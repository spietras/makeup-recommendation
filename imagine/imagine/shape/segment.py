from abc import ABC

import numpy as np
from sklearn.base import clone

from imagine.functional.functional import ImageOperation, Batchable


class Segmenter(ImageOperation, ABC):
    """Segment images into parts"""

    def __init__(self, org_parts_map, parts_map=None, bg_code=0):
        """
        Args:
            org_parts_map: original map from part names to codes
            parts_map: target map from parts names to codes. if None no remapping is done and original map is used.
            bg_code: code to use for background. defaults to 0.
        """

        super().__init__()
        self.org_parts_map = org_parts_map
        self.parts_map = parts_map
        self.bg_code = bg_code

    def __call__(self, imgs, **kwargs):
        """
        Perform segmenting on images

        Args:
            imgs: numpy array of shape ([N], height, width, 3) with values adjusted for segmenting action

        Returns:
            numpy array of shape ([N], height, width, 1)
        """
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
    """Segmenting using face parser. Batchable. Use RGB values."""

    def __init__(self, parser, parts_map=None, bg_code=0):
        super().__init__({p: c for c, p in parser.codes.items()}, parts_map, bg_code)
        self.parser = parser

    def perform(self, imgs, masks=None, **kwargs):
        parsed = self.parser.parse(imgs)
        if masks is not None:
            parsed[masks == 0] = self.bg_code
        return parsed


class ClusteringSegmenter(Segmenter):
    """Segmenting using clustering. Use any values, but Lab colorspace is recommended."""

    class IdentityDict(dict):
        def __getitem__(self, k):
            return k

    def __init__(self, clustering, ordering=lambda labels, pixels: list(range(max(labels) + 1)), parts_map=None, bg_code=0):
        """
        Args:
            clustering: sklearn clustering algorithm with fit_predict() method
            ordering: Function of number of clusters, labels given to pixels and pixel values that should return iterable
                      with labels order. Defaults to numerical ordering.
        """

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
