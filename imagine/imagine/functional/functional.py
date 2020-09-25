from abc import ABC, abstractmethod
from collections import Iterable

import numpy as np


class Callable(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        return NotImplemented

    def execute(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    @classmethod
    def withArgs(cls, **outer_kwargs):
        class CallableWithArgs(cls):
            def __init__(self, callable_kwargs):
                self.callable_kwargs = callable_kwargs

            def __call__(self, *args, **kwargs):
                target_kwargs = {k: v(*args, **kwargs) for k, v in self.callable_kwargs.items()}
                target_kwargs.update(kwargs)
                return super().__call__(*args, **target_kwargs)

        return CallableWithArgs(outer_kwargs)

    @classmethod
    def asArray(cls):
        class CallableAsArray(cls):
            def __call__(self, *args, **kwargs):
                return np.asarray(super().__call__(*args, **kwargs))

        return CallableAsArray()


class SinglePositionalArgCallable(Callable, ABC):
    @abstractmethod
    def __call__(self, x, **kwargs):
        return NotImplemented


class BatchOperation(SinglePositionalArgCallable, ABC):

    def __call__(self, x, **kwargs):
        if self.is_batch(x) and not self.is_batchable():
            args = [dict([(k, v[i]) for k, v in kwargs.items()]) for i, _ in enumerate(x)]
            results = [self.perform(i, **a) for i, a in zip(x, args)]
            return self.stack(results)
        if not self.is_batch(x) and self.is_batchable():
            x_expanded = self.expand(x)
            args_expended = {k: self.expand(v) for k, v in kwargs.items()}
            return self.squeeze(self.perform(x_expanded, **args_expended))
        return self.perform(x, **kwargs)

    @staticmethod
    def is_batch(x):
        return isinstance(x, Iterable) and not isinstance(x, str)

    @staticmethod
    def is_batchable():
        return False

    @staticmethod
    def stack(results):
        return results

    @staticmethod
    def expand(x):
        return [x]

    @staticmethod
    def squeeze(results):
        return results[0]

    @abstractmethod
    def perform(self, x, **kwargs):
        return NotImplemented


class Batchable(BatchOperation, ABC):
    @staticmethod
    def is_batchable():
        return True


class ImageOperation(BatchOperation, ABC):

    @staticmethod
    def is_batch(img):
        return img.ndim == 4

    def stack(self, results):
        return np.stack(results)

    def expand(self, img):
        return np.expand_dims(img, 0)

    @abstractmethod
    def perform(self, img, **kwargs):
        """
        Perform operation on image or sequence of images

        Args:
            img: numpy array with dimensions ([N], width, height, channels) with image data adjusted for this particular operation

        Returns:
            input after performing the operation on it
        """
        return NotImplemented


class Identity(SinglePositionalArgCallable):
    def __call__(self, x, **kwargs):
        return x


class Constant(BatchOperation):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def perform(self, x, **kwargs):
        return self.constant


class Channelize(SinglePositionalArgCallable):
    def __call__(self, x, **kwargs):
        return x[..., np.newaxis]


class Dechannelize(SinglePositionalArgCallable):
    def __call__(self, x, **kwargs):
        return x[..., -1]


class Lambda(SinglePositionalArgCallable):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def __call__(self, x, **kwargs):
        return self.f(x, **kwargs)


class Join(SinglePositionalArgCallable):
    def __init__(self, operations):
        super().__init__()
        self.operations = operations

    def __call__(self, x, **kwargs):
        for operation in self.operations:
            x = operation(x, **kwargs)
        return x
