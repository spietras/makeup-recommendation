# now you can import that way:
# from package import module.foo
# or it it's exposed in __init__.py:
# from package import foo

import argparse

from automakeup import recommenders
from webmakeup import workers
from webmakeup.server import Server
from webmakeup.server import DEFAULT_PORT


def parse_args():
    parser = argparse.ArgumentParser(prog=__package__, description="{} - automakeup server".format(__package__))
    parser.add_argument('port', nargs='?', type=int, default=DEFAULT_PORT, help='port at which to run the server')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    server = None

    try:
        recommender = recommenders.DummyRecommender()
        worker = workers.MakeupWorker(recommender)
        server = Server(worker, args.port)
    except IOError as e:
        print("Can't load server. Error occurred: ", e)
        exit(1)

    try:
        server.run()
    finally:
        server.cleanup()
