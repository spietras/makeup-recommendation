from abc import ABC, abstractmethod

import numpy as np

from imagine.helpers import normalize_photo

DEFAULT_PARTS_MAP = {
    "face": 1,
    "lips": 2,
    "left_eye": 3,
    "right_eye": 4,
    "hair": 5
}

RECOGNIZED_PARTS = DEFAULT_PARTS_MAP.keys()


class Segmenter(ABC):
    @abstractmethod
    def segment(self, imgs, parts_map=None, bg_code=0):
        """
        Segment images into parts

        Args:
            img - numpy array of shape (N, height, width, 3) in RGB with values either in [0-255] or [0.0-1.0]
            parts_map - dict with mapping of part name to code to use

        Returns:
            numpy array of shape (N, height, width) with values of codes in pixels recognized as parts
        """
        return NotImplemented


class FaceParsingSegmenter(Segmenter):

    def __init__(self, parser):
        self.parser = parser
        self.parts_name_map = {
            "skin": "face",
            "u_lip": "lips",
            "l_lip": "lips",
            "l_eye": "left_eye",
            "r_eye": "right_eye",
            "hair": "hair"
        }

    def segment(self, img, parts_map=None, bg_code=0):
        if not parts_map:
            parts_map = DEFAULT_PARTS_MAP

        img = normalize_photo(img)

        parsed = self.parser.parse(img)
        return self._remap(parsed, parts_map, bg_code)

    def _remap(self, parsed, parts_map, bg_code):
        remapped = np.full(parsed.shape, bg_code)
        part_to_code = {v: k for k, v in self.parser.codes.items()}
        for p in self.parts_name_map:
            if self.parts_name_map[p] in parts_map:
                remapped[parsed == part_to_code[p]] = parts_map[self.parts_name_map[p]]
        return remapped
