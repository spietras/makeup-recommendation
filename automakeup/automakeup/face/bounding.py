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
        if len(bbs) == 0:
            return None
        biggest_bb = max(bbs, key=lambda rect: rect.width() * rect.height())
        return Rect.from_dlib(biggest_bb)


class MTCNNBoundingBoxFinder(BoundingBoxFinder):
    def __init__(self, mtcnn):
        super().__init__()
        self.mtcnn = mtcnn

    def find(self, img):
        bbs, _ = self.mtcnn.find(np.expand_dims(img, 0))
        if bbs[0] is None or bbs[0].size == 0:
            return None
        best_face_bb = bbs[0][0]
        return Rect(best_face_bb[1], best_face_bb[3], best_face_bb[0], best_face_bb[2])
