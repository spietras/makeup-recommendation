from abc import ABC, abstractmethod
from collections import OrderedDict

from sklearn.base import BaseEstimator
from torch import nn


class LoadableModule(nn.Module):
    def __init__(self, *params):
        super().__init__()
        self.params = list(params)

    @classmethod
    def load(cls, state_dict, **tunable_params):
        if "model" in state_dict and "params" in state_dict:
            model_state_dict = state_dict["model"]
            params = state_dict["params"]
        else:
            model_state_dict = state_dict
            params = []

        model = cls(*params, **tunable_params)
        model.load_state_dict(model_state_dict)
        return model

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        model_state_dict = super().state_dict(None, prefix, keep_vars)

        if len(self.params) == 0:
            destination.update(model_state_dict)
        else:
            destination[prefix + "model"] = model_state_dict
            destination[prefix + "params"] = self.params
        return destination


class ConditionalGenerativeModel(BaseEstimator, ABC):
    @abstractmethod
    def fit(self, x, y):
        return NotImplemented

    @abstractmethod
    def sample(self, y, state):
        return NotImplemented

    @abstractmethod
    def score(self, x, y):
        return NotImplemented
