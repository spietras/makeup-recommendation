from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import clone


class Segmenter(ABC):

    def __init__(self, org_parts_map):
        super().__init__()
        self.org_parts_map = org_parts_map

    def segment(self, imgs, parts_map=None, masks=None, bg_code=0):
        """
        Segment images into parts

        Args:
            imgs - numpy array of shape (N, height, width, C) with values appropriate for the segmenter
            parts_map - dict with mapping of part name to code to use
            masks - numpy array of shape (N, height, width) with ones or Trues in pixels to process (used before processing where possible, otherwise after)
            bg_code - code to fill background with

        Returns:
            numpy array of shape (N, height, width) with values of codes in pixels recognized as parts
        """
        if masks is None:
            masks = np.ones(imgs.shape[:-1])

        parsed = self._parse(imgs, masks, bg_code)
        if parts_map:
            return self._remap(parsed, parts_map, bg_code)
        return parsed

    @abstractmethod
    def _parse(self, imgs, masks, bg_code):
        return NotImplemented

    def _remap(self, parsed, new_parts_map, bg_code):
        remapped = np.full(parsed.shape, bg_code)
        for p in new_parts_map:
            remapped[parsed == self.org_parts_map[p]] = new_parts_map[p]
        return remapped


class BatchSegmenter(Segmenter, ABC):

    def _parse(self, imgs, masks, bg_code):
        parsed = self._parse_batch(imgs)
        if masks is not None:
            parsed[masks == 0] = bg_code
        return parsed

    @abstractmethod
    def _parse_batch(self, imgs):
        return NotImplemented


class SingleSegmenter(Segmenter, ABC):

    def _parse(self, imgs, masks, bg_code):
        return np.stack([self._parse_single(i, m, bg_code) for i, m in zip(imgs, masks)])

    @abstractmethod
    def _parse_single(self, img, mask, bg_code):
        return NotImplemented


class ParsingSegmenter(BatchSegmenter):

    def __init__(self, parser):
        super().__init__({p: c for c, p in parser.codes.items()})
        self.parser = parser

    def _parse_batch(self, imgs):
        return self.parser.parse(imgs)


class ClusteringSegmenter(SingleSegmenter):
    class IdentityDict(dict):
        def __getitem__(self, k):
            return k

    def __init__(self, clustering, ordering=lambda labels, pixels: np.unique(labels)):
        super().__init__(self.IdentityDict())
        self.clustering = clustering
        self.ordering = ordering

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
