import json
import logging
from abc import ABC, abstractmethod
from typing import BinaryIO

import cv2
import numpy as np

from imagine.color.conversion import BgrToRgb

logger = logging.getLogger("workers")


class Worker(ABC):
    @abstractmethod
    def work(self, *args):
        """
        Implement this method to provide your endpoint logic

        Args:
            You can provide arguments to control the required endpoint parameters
            If your implementation looks like this:

            def work(self, foo, bar)

            then the server will require foo and bar parameters from the clients

            By default all parameters are passed as strings.
            However you can use type hints to enforce conversion.
            For example:

            def work(self, foo : int, bar : BinaryIO)

            will ensure that passed foo parameter is a string and bar is a BinaryIO (like an image).

        Returns:
            Dictionary with parameters names and values. For example:

            {
                'foo' : 1,
                'bar' : 'foobar'
            }
        """
        return NotImplemented

    def cleanup(self):
        pass


class MakeupWorker(Worker):
    class SimpleEncoder(json.JSONEncoder):
        def default(self, o):
            return o.__dict__

    def __init__(self, pipeline, encoder=SimpleEncoder):
        self.pipeline = pipeline
        self.encoder = encoder

    @staticmethod
    def stream_to_rgb(input):
        array = np.frombuffer(input.read(), dtype=np.uint8)
        return BgrToRgb(cv2.imdecode(array, cv2.IMREAD_COLOR))

    def work(self, img: BinaryIO):
        try:
            img_rgb = self.stream_to_rgb(img)
        except Exception as e:
            logger.warning("Exception occurred during image parameter conversion", exc_info=e)
            raise ValueError("Can't convert parameter to image")
        return json.loads(json.dumps(self.pipeline.run(img_rgb), cls=self.encoder))
