import argparse
import contextlib
import io
import sys

from automakeup import recommenders


def parse_args():
    parser = argparse.ArgumentParser(prog=__package__, description="{} - automakeup cli".format(__package__))
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


if __name__ == '__main__':
    args = parse_args()

    recommender = recommenders.DummyRecommender()

    with open_input(args.file) as i:
        print(recommender.recommend(i))
