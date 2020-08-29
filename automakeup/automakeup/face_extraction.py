from abc import ABC, abstractmethod

import openface


class FaceExtractor(ABC):
    @abstractmethod
    def extract(self, input):
        return NotImplemented


class OpenFaceExtractor(FaceExtractor):
    def extract(self, input):
        align = openface.AlignDlib(args.dlibFacePredictor)
