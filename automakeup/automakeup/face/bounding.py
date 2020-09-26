from abc import ABC, abstractmethod

import dlib
import numpy as np

from imagine.shape.figures import Rect


class BoundingBoxFinder(ABC):
    @abstractmethod
    def find(self, img):
        return NotImplemented


class DlibBoundingBoxFinder(BoundingBoxFinder):
    def __init__(self):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()

    def find(self, img):
        bbs = self.detector(img, 1)
        biggest_bb = max(bbs, key=lambda rect: rect.width() * rect.height())
        return Rect.from_dlib(biggest_bb)


class MTCNNBoundingBoxFinder(BoundingBoxFinder):
    def __init__(self, mtcnn):
        super().__init__()
        self.mtcnn = mtcnn

    def find(self, img):
        bbs, _ = self.mtcnn.find(np.expand_dims(img, 0))
        biggest_bb = bbs[0][0]
        return Rect(biggest_bb[1], biggest_bb[3], biggest_bb[0], biggest_bb[2])
