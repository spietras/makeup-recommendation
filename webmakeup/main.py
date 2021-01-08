import importlib.resources as pkg_resources
import logging

import configargparse
import torch

from automakeup.pipelines import GanettePipeline
from webmakeup.server import Server, DEFAULT_HOST, DEFAULT_PORT
from workers import MakeupWorker


def parse_args():
    with pkg_resources.path("resources", "config.yaml") as config_path:
        argparser = configargparse.ArgParser(prog=__package__,
                                             description="{} - automakeup server".format(__package__),
                                             default_config_files=[str(config_path)])
        argparser.add_argument('--host', type=str, default=DEFAULT_HOST, help='host at which to run the server')
        argparser.add_argument('--port', type=int, default=DEFAULT_PORT, help='port at which to run the server')
    return argparser.parse_args()


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def config_logging():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')


if __name__ == '__main__':
    args = parse_args()
    device = get_device()
    config_logging()
    server = None

    logger = logging.getLogger("main")
    logger.info("Using device = {}".format(str(device)))

    try:
        logger.info("Loading pipeline...")
        pipeline = GanettePipeline(device=device)
        logger.info("Pipeline loaded")
        worker = MakeupWorker(pipeline)
        server = Server(worker, args.host, args.port)
    except IOError as e:
        logger.error("Can't load server", exc_info=e)
        exit(1)

    logger.info("Server loaded. Running...")
    try:
        server.run()
    finally:
        logger.info("Cleaning up...")
        server.cleanup()
