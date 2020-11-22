import argparse

from automakeup import recommenders
from webmakeup import workers
from webmakeup.server import Server, DEFAULT_HOST, DEFAULT_PORT


def parse_args():
    parser = argparse.ArgumentParser(prog=__package__, description="{} - automakeup server".format(__package__))
    parser.add_argument('--host', type=str, default=DEFAULT_HOST, help='host at which to run the server')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='port at which to run the server')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    server = None

    try:
        recommender = recommenders.DummyRecommender()
        worker = workers.MakeupWorker(recommender)
        server = Server(worker, args.host, args.port)
    except IOError as e:
        print("Can't load server. Error occurred: ", e)
        exit(1)

    try:
        server.run()
    finally:
        server.cleanup()
