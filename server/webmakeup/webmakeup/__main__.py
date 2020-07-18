# now you can import that way:
# from package import module.foo
# or it it's exposed in __init__.py:
# from package import foo

import argparse

from webmakeup import package_name
from webmakeup import workers
from webmakeup.server import DEFAULT_PORT
from webmakeup.server import Server


def parse_args():
    parser = argparse.ArgumentParser(description="{} - automakeup server".format(package_name))
    parser.add_argument('port', nargs='?', type=int, default=DEFAULT_PORT)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    server = None

    try:
        worker = workers.MakeupWorker()
        server = Server(worker, args.port)
    except IOError as e:
        print("Can't load server. Error occurred: ", e)
        exit(1)

    try:
        server.run()
    finally:
        server.cleanup()
