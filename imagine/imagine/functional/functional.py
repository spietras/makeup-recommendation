from abc import ABC, abstractmethod
from collections import Iterable

import numpy as np


class BatchClassifier(ABC):
    @staticmethod
    @abstractmethod
    def is_batch(x):
        """Returns true if x should be considered as batch"""
        return NotImplemented


class SimpleBatchClassifier(BatchClassifier):
    """All iterables except string are batches"""

    @staticmethod
    def is_batch(x):
        return isinstance(x, Iterable) and not isinstance(x, str)


class ImageBatchClassifier(BatchClassifier):
    """Only numpy arrays of 4 dimensions are batches"""

    @staticmethod
    def is_batch(x):
        return x.ndim == 4


class Callable(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        return NotImplemented

    def execute(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def withArgs(self, **outer_kwargs):
        """
        Use to pass additional arguments to perform action at call time

        Examples:
            class Dummy(Callable):
                def __call__(self, *args, arg=None, **kwargs):
                    return arg

            op = Dummy().withArgs(arg=Constant(5))

        Args:
            **outer_kwargs: dict with keys with names of target action arguments and values as Callables

        Returns:
            new Callable that will pass arguments to target action evaluating each arguments with passed Callable
        """

        class CallableWithArgs(Callable):
            def __init__(self, inner, callable_kwargs):
                self.inner = inner
                self.callable_kwargs = callable_kwargs

            def __call__(self, *args, **kwargs):
                target_kwargs = {k: v(*args, **kwargs) for k, v in self.callable_kwargs.items()}
                target_kwargs.update(kwargs)
                return self.inner.__call__(*args, **target_kwargs)

        return CallableWithArgs(self, outer_kwargs)

    def asArray(self):
        """Return results as numpy array"""

        class CallableAsArray(Callable):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def __call__(self, *args, **kwargs):
                return np.asarray(self.inner.__call__(*args, **kwargs))

        return CallableAsArray(self)


class SinglePositionalArgCallable(Callable, ABC):
    @abstractmethod
    def __call__(self, x, **kwargs):
        return NotImplemented


class BatchOperation(SinglePositionalArgCallable, ABC):
    """Callable that detects if input is batch and adjust it for action needs"""

    def __init__(self, batch_classifier=SimpleBatchClassifier()):
        """
        Args:
            batch_classifier: BatchClassifier object which will be used to tell is input is batch or not
        """
        super().__init__()
        self.batch_classifier = batch_classifier

    def __call__(self, x, **kwargs):
        if self.batch_classifier.is_batch(x) and not self.is_batchable():
            args = [dict([(k, v[i]) for k, v in kwargs.items()]) for i, _ in enumerate(x)]
            results = [self.perform(i, **a) for i, a in zip(x, args)]
            return self.stack(results)
        if not self.batch_classifier.is_batch(x) and self.is_batchable():
            x_expanded = self.expand(x)
            args_expended = {k: self.expand(v) for k, v in kwargs.items()}
            return self.squeeze(self.perform(x_expanded, **args_expended))
        return self.perform(x, **kwargs)

    @staticmethod
    def is_batchable():
        """Override to mark your action as batchable"""

        return False

    @staticmethod
    def stack(results):
        """Override to provide stacking (collecting results from individual samples) behaviour adjusted for your action"""
        return results

    @staticmethod
    def expand(x):
        """Override to provide expanding (making singular batch from single sample) behaviour adjusted for your action"""
        return [x]

    @staticmethod
    def squeeze(results):
        """Override to provide squeezing (extracting single sample from singular batch) behaviour adjusted for your action"""
        return results[0]

    @abstractmethod
    def perform(self, x, **kwargs):
        """
        Override to provide your action behaviour.
        x is single sample or batch depending on your action being batchable or not.
        """
        return NotImplemented


class Batchable(BatchOperation, ABC):
    """Mixin marking your action as batchable"""

    @staticmethod
    def is_batchable():
        return True


class ImageOperation(BatchOperation, ABC):
    """Operation treating single sample as numpy array of shape (height, width, channels)"""

    def __init__(self):
        super().__init__(ImageBatchClassifier())

    def stack(self, results):
        return np.stack(results)

    def expand(self, img):
        return np.expand_dims(img, 0)

    @abstractmethod
    def perform(self, img, **kwargs):
        """
        Perform operation on image or sequence of images

        Args:
            img: numpy array with dimensions ([N], height, width, channels) with image data adjusted for this particular operation

        Returns:
            input after performing the operation on it
        """
        return NotImplemented


class Identity(SinglePositionalArgCallable):
    """Operation returning the input"""

    def __call__(self, x, **kwargs):
        return x


class Constant(BatchOperation):
    """Returns constant for single sample"""

    def __init__(self, constant, batch_classifier=SimpleBatchClassifier()):
        super().__init__(batch_classifier)
        self.constant = constant

    def perform(self, x, **kwargs):
        return self.constant


class Channelize(SinglePositionalArgCallable):
    """Adds singular channel dimension"""

    def __call__(self, x, **kwargs):
        return x[..., np.newaxis]


class Dechannelize(SinglePositionalArgCallable):
    """Removes singular channel dimension"""

    def __call__(self, x, **kwargs):
        return x[..., -1]


class Lambda(BatchOperation):
    """Operation with custom logic for single sample"""

    def __init__(self, f, batch_classifier=SimpleBatchClassifier()):
        super().__init__(batch_classifier)
        self.f = f

    def perform(self, x, **kwargs):
        return self.f(x, **kwargs)


class Join(SinglePositionalArgCallable):
    """Joins multiple operations pipe-like"""

    def __init__(self, operations):
        super().__init__()
        self.operations = operations

    def __call__(self, x, **kwargs):
        for operation in self.operations:
            x = operation(x, **kwargs)
        return x
