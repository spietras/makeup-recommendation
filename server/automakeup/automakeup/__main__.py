# now you can import that way:
# from package import module.foo
# or it it's exposed in __init__.py:
# from package import foo

import argparse
import contextlib
import io
import sys

from automakeup import package_name
from automakeup import recommenders


def parse_args():
    parser = argparse.ArgumentParser(prog=package_name,
                                     description="{} - automatic makeup recommendation".format(package_name))
    parser.add_argument('file', nargs='?', default='-',
                        help="path to the image file with a face (when not given, reads content directly from stdin)")
    return parser.parse_args()


@contextlib.contextmanager
def open_input(filename: str):
    if filename == '-':
        yield io.BytesIO(sys.stdin.buffer.read())
    else:
        with open(filename, 'rb') as f:
            yield f


if __name__.endswith('__main__'):
    args = parse_args()

    recommender = recommenders.DummyRecommender()

    with open_input(args.file) as i:
        print(recommender.recommend(i))
