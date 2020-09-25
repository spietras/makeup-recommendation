from abc import ABC, abstractmethod

from imagine.shape.figures import Rect


class BoundingBoxFinder(ABC):
    @abstractmethod
    def find(self, img):
        return NotImplemented


class OpenfaceBoundingBoxFinder(BoundingBoxFinder):
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def find(self, img):
        bb = self.estimator.getLargestFaceBoundingBox(img)
        return Rect.from_dlib(bb)
