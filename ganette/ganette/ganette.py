import contextlib
import itertools
import os

import numpy as np
import pykeops
import torch
from geomloss import SamplesLoss
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array, check_consistent_length
from torch import nn, optim
from torch.autograd import grad

from modelutils import ConditionalGenerativeModel, LearningLogger, Picklable, LoadableModule


class Ganette(ConditionalGenerativeModel, BaseEstimator, Picklable):
    class Generator(LoadableModule):
        def __init__(self, x_features, y_features, latent_size, layers):
            super().__init__(x_features, y_features, latent_size, layers)

            layer_sizes = np.linspace(latent_size + y_features, x_features, layers + 1).astype(np.int)

            layers = list(itertools.chain.from_iterable(
                [(nn.Linear(s1, s2), nn.LeakyReLU()) for s1, s2 in zip(layer_sizes, layer_sizes[1:])]
            ))[:-1]

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    class Discriminator(nn.Module):
        def __init__(self, x_features, y_features, layers, dropout_prob):
            super().__init__()

            layer_sizes = np.linspace(x_features + y_features, 1, layers + 1).astype(np.int)

            layers = list(itertools.chain.from_iterable(
                [(nn.Linear(s1, s2), nn.LeakyReLU(), nn.Dropout(dropout_prob)) for s1, s2 in
                 zip(layer_sizes, layer_sizes[1:])]
            ))[:-2]

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    def __init__(self,
                 *,
                 generator_n_layers=4,
                 discriminator_n_layers=4,
                 latent_size=4,
                 discriminator_dropout_prob=0.03,
                 generator_lr=0.0001,
                 discriminator_lr=0.0004,
                 gp_lambda=10,
                 device=torch.device('cpu'),
                 batch_size=4,
                 shuffle=True,
                 epochs=50,
                 score_sample_times=1,
                 random_state=None):
        super().__init__()
        self.generator_n_layers = generator_n_layers
        self.discriminator_n_layers = discriminator_n_layers
        self.latent_size = latent_size
        self.discriminator_dropout_prob = discriminator_dropout_prob
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.gp_lambda = gp_lambda
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        self.score_sample_times = score_sample_times
        self.random_state = random_state

    def _more_tags(self):
        return {'requires_y': True}

    def _validate_array_param(self, input, param_name, reset=False):
        input = check_array(input)
        dtype, n_features = input.dtype, input.shape[1]
        dtype_attrib_name, features_attrib_name = f"{param_name}_dtype_", f"{param_name}_n_features_in_"
        if reset:
            self.__setattr__(dtype_attrib_name, dtype)
            self.__setattr__(features_attrib_name, n_features)
        expected_features = self.__getattribute__(features_attrib_name)
        if n_features != expected_features:
            raise ValueError(
                f"{param_name} has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {expected_features} features as input.")
        return input

    def _validate_x(self, x, reset=False):
        return self._validate_array_param(x, "x", reset)

    def _validate_y(self, x, reset=False):
        return self._validate_array_param(x, "y", reset)

    def fit(self, x, y):
        x, y = self._validate_x(x, reset=True), self._validate_y(y, reset=True)
        check_consistent_length(x, y)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        x = torch.as_tensor(x, device=self.device, dtype=torch.float)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float)
        x_features, y_features = x.shape[1], y.shape[1]
        g = self.Generator(x_features, y_features,
                           self.latent_size, self.generator_n_layers).to(self.device)
        d = self.Discriminator(x_features, y_features,
                               self.discriminator_n_layers, self.discriminator_dropout_prob).to(self.device)
        g_optimizer = optim.Adam(g.parameters(), lr=self.generator_lr)
        d_optimizer = optim.Adam(d.parameters(), lr=self.discriminator_lr)

        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        self.logger_ = LearningLogger()

        for _ in range(self.epochs):
            d_total_loss, g_total_loss = 0, 0
            for x, y in loader:
                d_optimizer.zero_grad()
                d_loss = self._loss_d(d, g, x, y)
                d_total_loss += d_loss.item()
                d_loss.backward()
                d_optimizer.step()

                g_optimizer.zero_grad()
                g_loss = self._loss_g(g, d, x, y)
                g_total_loss += g_loss.item()
                g_loss.backward()
                g_optimizer.step()

            self.logger_.log(d_total_loss / len(loader), "d_loss")
            self.logger_.log(g_total_loss / len(loader), "g_loss")

        self.g_ = g
        return self

    def _loss_d(self, d, g, x, y):
        n_samples = len(x)

        d_in_real = torch.cat([x, y], dim=1)

        z = torch.randn(n_samples, self.latent_size, device=self.device, dtype=x.dtype)
        g_in = torch.cat([z, y], dim=1)
        d_in_fake = torch.cat([g(g_in), y], dim=1)

        real_loss, fake_loss = d(d_in_real).mean(), d(d_in_fake).mean()
        gradient_penalty = self._gradient_penalty(d, d_in_real, d_in_fake)

        return fake_loss - real_loss + self.gp_lambda * gradient_penalty

    def _gradient_penalty(self, d, real, fake):
        n_samples = len(real)
        alpha = torch.rand(n_samples, 1, device=self.device, dtype=real.dtype)
        interpolations = torch.lerp(real, fake, alpha)
        loss = d(interpolations).mean()
        return (grad(loss, interpolations, create_graph=True)[0].norm() - 1).pow(2)

    def _loss_g(self, g, d, x, y):
        n_samples = len(x)

        z = torch.randn(n_samples, self.latent_size, device=self.device, dtype=x.dtype)
        g_in = torch.cat([z, y], dim=1)
        d_in = torch.cat([g(g_in), y], dim=1)

        return -d(d_in).mean()

    def sample(self, y, state=None):
        check_is_fitted(self)
        y = self._validate_y(y)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float)
        n_samples = len(y)
        rng = torch.Generator(device=self.device)
        if state is not None:
            rng.manual_seed(state)
        g_in = torch.cat([
            torch.randn(n_samples, self.latent_size, device=self.device, dtype=y.dtype, generator=rng),
            y], dim=1)
        return self.g_(g_in).detach().cpu().numpy().astype(self.x_dtype_)

    def score(self, x, y):
        check_is_fitted(self)
        x, y = self._validate_x(x), self._validate_y(y)
        check_consistent_length(x, y)
        pykeops.config.gpu_available = False
        Loss = SamplesLoss("sinkhorn", p=1, blur=.005, scaling=.9, backend="online")

        xy_fake = torch.as_tensor(
            np.vstack([np.hstack([self.sample(y), y]) for _ in range(self.score_sample_times)]),
            dtype=torch.float
        )
        xy = torch.as_tensor(np.hstack([x, y]), dtype=torch.float)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            dist = Loss(xy_fake.cpu().contiguous(), xy.cpu().contiguous()).item()
        return -dist

    def __getstate__(self):
        state = super().__getstate__()
        state["g_"] = self.g_.state_dict()
        return state

    def __setstate__(self, state):
        self.g_ = self.Generator.load(state["g_"])
        state.pop("g_")
        super().__setstate__(state)
