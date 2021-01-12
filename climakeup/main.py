import contextlib
import importlib.resources as pkg_resources
import io
import json
import sys

import configargparse
import cv2
import numpy as np
import torch

from automakeup.pipelines import GanettePipeline
from imagine.color.conversion import BgrToRgb


def parse_args():
    with pkg_resources.path("resources", "config.yaml") as config_path:
        argparser = configargparse.ArgParser(prog=__package__,
                                             description="{} - automakeup command line interface".format(__package__),
                                             default_config_files=[str(config_path)])
        argparser.add_argument('file', nargs='?', default='-',
                               help="path to the image file with a face (when not given, reads content directly from stdin)")
        argparser.add_argument('--config', is_config_file=True,
                               help='config file path')
    return argparser.parse_args()


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@contextlib.contextmanager
def open_input(filename: str):
    if filename == '-':
        yield io.BytesIO(sys.stdin.buffer.read())
    else:
        with open(filename, 'rb') as f:
            yield f


def get_image(input):
    array = np.frombuffer(input.read(), dtype=np.uint8)
    return BgrToRgb(cv2.imdecode(array, cv2.IMREAD_COLOR))


if __name__ == '__main__':
    args = parse_args()
    device = get_device()

    with open_input(args.file) as i:
        result = GanettePipeline(device=device).run(get_image(i))

    print(json.dumps(result.__dict__, indent=4))
