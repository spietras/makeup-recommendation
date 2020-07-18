from flask import Flask

from webmakeup import package_name
from webmakeup.handlers import EndpointHandler

FLASK_PRETTYPRINT_OPTION_NAME = 'JSONIFY_PRETTYPRINT_REGULAR'
DEFAULT_PORT = 8080


class Server:
    def __init__(self, worker, port=DEFAULT_PORT, pretty_print=True):
        self.worker = worker
        self.port = port
        self.app = self._get_flask_app(package_name, pretty_print)

    def _get_flask_app(self, name, pretty_print):
        def add_endpoint(app, route, f):
            app.add_url_rule(route, f.__name__, EndpointHandler(f))

        app = Flask(name)
        app.config[FLASK_PRETTYPRINT_OPTION_NAME] = pretty_print
        add_endpoint(app, "/", self.worker.work)
        return app

    def run(self):
        self.app.run(port=self.port)

    def cleanup(self):
        self.worker.cleanup()
