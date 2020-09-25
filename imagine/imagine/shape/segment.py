from abc import ABC

import numpy as np
from sklearn.base import clone

from imagine.functional.functional import ImageOperation, Batchable


class Segmenter(ImageOperation, ABC):
    def __init__(self, org_parts_map, parts_map=None, bg_code=0):
        super().__init__()
        self.org_parts_map = org_parts_map
        self.parts_map = parts_map
        self.bg_code = bg_code

    def __call__(self, imgs, **kwargs):
        parsed = super().__call__(imgs, **kwargs)
        if self.parts_map:
            return self._remap(parsed, self.parts_map, self.bg_code)
        return parsed

    def _remap(self, parsed, new_parts_map, bg_code):
        remapped = np.full(parsed.shape, bg_code)
        for p in new_parts_map:
            remapped[parsed == self.org_parts_map[p]] = new_parts_map[p]
        return remapped


class ParsingSegmenter(Batchable, Segmenter):
    def __init__(self, parser, parts_map=None, bg_code=0):
        super().__init__({p: c for c, p in parser.codes.items()}, parts_map, bg_code)
        self.parser = parser

    def perform(self, imgs, masks=None, **kwargs):
        parsed = self.parser.parse(imgs)
        if masks is not None:
            parsed[masks == 0] = self.bg_code
        return parsed


class ClusteringSegmenter(Segmenter):
    class IdentityDict(dict):
        def __getitem__(self, k):
            return k

    def __init__(self, clustering, ordering=lambda labels, pixels: np.unique(labels), parts_map=None, bg_code=0):
        super().__init__(self.IdentityDict(), parts_map, bg_code)
        self.clustering = clustering
        self.ordering = ordering

    def perform(self, img, masks=None, **kwargs):
        index = masks == 1 if masks is not None else np.ones(img.shape[:-1]) == 1
        pixels = img[index]
        if pixels.size:
            clustered = clone(self.clustering).fit_predict(pixels)
            clustered = np.argsort(self.ordering(clustered, pixels))[clustered]
            parsed = np.full(img.shape[:-1], self.bg_code, dtype=clustered.dtype)
            parsed[index] = clustered
        else:
            parsed = np.full(img.shape[:-1], self.bg_code)
        return parsed

    def _parse_single(self, img, mask, bg_code):
        pixels = img[mask == 1]
        if pixels.size:
            clustered = clone(self.clustering).fit_predict(pixels)
            clustered = np.argsort(self.ordering(clustered, pixels))[clustered]
            parsed = np.full(img.shape[:-1], bg_code, dtype=clustered.dtype)
            parsed[mask == 1] = clustered
        else:
            parsed = np.full(img.shape[:-1], bg_code)
        return parsed
