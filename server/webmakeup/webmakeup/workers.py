from abc import ABC, abstractmethod
from typing import BinaryIO


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
    def __init__(self, recommender):
        self.recommender = recommender

    def work(self, img: BinaryIO):
        return self.recommender.recommend(img)
