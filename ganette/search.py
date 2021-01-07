import contextlib
import importlib.resources as pkg_resources
import json
import logging
import pickle

import configargparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from ganette import Ganette
from imagine.color import conversion
from imagine.functional import functional as fun


def parse_args():
    with pkg_resources.path("resources", "config.yaml") as config_path:
        argparser = configargparse.ArgParser(prog=__package__,
                                             description="{} - Ganette best model search".format(__package__),
                                             default_config_files=[str(config_path)])
        argparser.add_argument("data",
                               help="path to data input file")
        argparser.add_argument("model_output",
                               help="path to the model output file")
        argparser.add_argument("params_output",
                               help="path to the params output file")
        argparser.add_argument("x_scaler_output",
                               help="path to the x scaler output file")
        argparser.add_argument("y_scaler_output",
                               help="path to the y scaler output file")
        argparser.add_argument('--config', is_config_file=True,
                               help='config file path')
        argparser.add_argument("--generator_n_layers", nargs="+", type=float,
                               help="Generator layer sizes to try")
        argparser.add_argument("--discriminator_n_layers", nargs="+", type=float,
                               help="Discriminator layer sizes to try")
        argparser.add_argument("--latent_size", nargs="+", type=float,
                               help="Latent sizes to try")
        argparser.add_argument("--discriminator_dropout_prob", nargs="+", type=float,
                               help="Discriminator dropout probabilities to try")
        argparser.add_argument("--generator_lr", nargs="+", type=float,
                               help="Generator learning rates to try")
        argparser.add_argument("--discriminator_lr", nargs="+", type=float,
                               help="Discriminator learning rates to try")
        argparser.add_argument("--gp_lambda", nargs="+", type=float,
                               help="Gradient penalty coefficients to try")
        argparser.add_argument("--batch_size", nargs="+", type=int,
                               help="Batch sizes to try")
        argparser.add_argument("--epochs", type=int, default=500,
                               help="number of epochs to train the model for in each iteration")
        argparser.add_argument("--n_iter", type=int, default=100,
                               help="number of iterations for random search")
        args = argparser.parse_args()
    return args


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def config_logging():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')


def get_data(path):
    df = pd.read_csv(path, na_values=-1).dropna().convert_dtypes()

    def make_lab_parts(df, parts):
        def make_lab(rgb):
            return fun.Join([
                fun.Rearrange("n c -> 1 n c"),
                conversion.RgbToLab,
                fun.Rearrange("1 n c -> n c")
            ])(rgb)

        return np.hstack([make_lab(df[[f"{p}_r", f"{p}_g", f"{p}_b"]].values.astype(np.uint8)) for p in parts])

    x_parts, y_parts = ["lipstick", "eyeshadow0", "eyeshadow1", "eyeshadow2"], ["skin", "hair", "lips", "eyes"]
    x, y = make_lab_parts(df, x_parts), make_lab_parts(df, y_parts)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x, y = x_scaler.fit_transform(x), y_scaler.fit_transform(y)
    return x, y, x_scaler, y_scaler


def get_params(args):
    return {
        "generator_n_layers": args.generator_n_layers or [2, 4, 6, 8, 10],
        "discriminator_n_layers": args.discriminator_n_layers or [2, 4, 6, 8, 10],
        "latent_size": args.latent_size or [2, 4, 6, 8, 10, 20, 40, 80],
        "discriminator_dropout_prob": args.discriminator_dropout_prob or uniform(loc=0, scale=0.1),
        "generator_lr": args.generator_lr or loguniform(0.00001, 0.01),
        "discriminator_lr": args.discriminator_lr or loguniform(0.00001, 0.01),
        "gp_lambda": args.gp_lambda or loguniform(0.1, 10),
        "batch_size": args.batch_size or [32, 64, 128, 256]
    }


@contextlib.contextmanager
def log_stdout(logger):
    logger.write = lambda msg: logger.info(msg) if msg != '\n' else None
    with contextlib.redirect_stdout(logger):
        yield None


def save_model(model, path, protocol=pickle.HIGHEST_PROTOCOL):
    with open(path, "wb") as f:
        model.pickle(f, protocol=protocol)


def save_params(params, path, indent=4):
    with open(path, "w") as f:
        json.dump(params, f, indent=indent)


def pickle_dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parse_args()
    device = get_device()
    config_logging()

    logger = logging.getLogger("search")
    logger.info("Using device = {}".format(str(device)))
    logger.info("Loading")

    x, y, x_scaler, y_scaler = get_data(args.data)

    base_model = Ganette(device=device, epochs=args.epochs)
    params_distributions = get_params(args)

    logger.info("Starting search")
    with log_stdout(logger):
        search = RandomizedSearchCV(base_model, params_distributions, n_iter=args.n_iter, verbose=3).fit(x, y)
    logger.info(f"Best score: {search.best_score_}")

    logger.info(f"Saving model to {args.model_output}")
    save_model(search.best_estimator_, args.model_output)

    logger.info(f"Saving params to {args.params_output}")
    save_params(search.best_params_, args.params_output)

    logger.info(f"Saving x scaler to {args.x_scaler_output}")
    pickle_dump(x_scaler, args.x_scaler_output)
    logger.info(f"Saving y scaler to {args.y_scaler_output}")
    pickle_dump(y_scaler, args.y_scaler_output)

    logger.info("Done")
