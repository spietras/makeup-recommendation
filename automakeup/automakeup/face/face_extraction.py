from abc import ABC, abstractmethod

import cv2
import dlib

from imagine.shape import operations


class FaceExtractor(ABC):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    @abstractmethod
    def extract(self, img, bb):
        return NotImplemented


class SimpleFaceExtractor(FaceExtractor):
    def __init__(self, output_size, bb_scale=2.0, interpolation=cv2.INTER_LINEAR):
        super().__init__(output_size)
        self.bb_scale = bb_scale
        self.interpolation = interpolation

    def extract(self, img, bb):
        bb = bb.scale(self.bb_scale)
        square_bb = operations.squarisize(bb)
        safe = operations.safe_rect(square_bb, img.shape, allow_scaling=True)
        cropped = operations.crop(img, safe)
        return operations.resize(cropped, (self.output_size, self.output_size), interpolation=self.interpolation)


class AligningDlibFaceExtractor(FaceExtractor):
    def __init__(self, output_size, predictor):
        super().__init__(output_size)
        self.predictor = predictor

    def extract(self, img, bb):
        dlib_bb = bb.to_dlib()
        landmarks = self.predictor(img, dlib_bb)
        return dlib.get_face_chip(img, landmarks, size=self.output_size)
