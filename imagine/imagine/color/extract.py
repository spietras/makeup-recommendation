from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import clone

from imagine.helpers import normalization
from imagine.color import conversion


class ColorExtractor(ABC):

    def extract(self, img, mask):
        """
        Extract colors from pixels

        Args:
            img - numpy array of shape (height, width, 3) in RGB with values either in [0-255] or [0.0-1.0]
            mask - numpy array of shape (height, width) with ones or Trues in pixels to process

        Returns:
            numpy array of shape (N, 3) with N extracted colors in RGB in [0-255] or None if mask is empty
        """
        img = normalization.ToUInt8()(img)
        img = conversion.RgbToLab(img)
        return self.extract_normalized(img, mask)

    @abstractmethod
    def extract_normalized(self, img, mask):
        return NotImplemented


class PositionAgnosticExtractor(ColorExtractor, ABC):

    def extract_normalized(self, img, mask):
        pixels = img[mask == 1]
        if pixels.size == 0:
            return np.empty((0, 3), dtype=np.uint8)
        extracted = self.extract_from_pixels(pixels)
        if len(extracted) == 0:
            return extracted
        return conversion.LabToRgb(np.expand_dims(extracted, 0))[0]

    @abstractmethod
    def extract_from_pixels(self, pixels):
        """
        Extract colors from flat pixel array

        Args:
            pixels - numpy array of shape (P, 3) in RGB with values either in [0-255]

        Returns:
            numpy array of shape (N, 3) with N extracted colors or None if mask is empty
        """
        return NotImplemented


class MeanColorExtractor(PositionAgnosticExtractor):
    """Color extractor based on mean pixel color"""

    def extract_from_pixels(self, pixels):
        return np.atleast_2d(pixels.mean(axis=0).astype(np.uint8))


class MedianColorExtractor(PositionAgnosticExtractor):
    """Color extractor based on median pixel color"""

    def extract_from_pixels(self, pixels):
        return np.atleast_2d(np.median(pixels, axis=0).astype(np.uint8))


class ClusteringColorExtractor(PositionAgnosticExtractor):
    """Color extractor based on clustering algorithm"""

    def __init__(self, clustering, ordering=lambda labels, pixels: list(range(max(labels) + 1))):
        """
        Args:
            clustering: sklearn clustering model with fit() method and labels_ attribute
        """
        super().__init__()
        self.clustering = clustering
        self.ordering = ordering

    def extract_from_pixels(self, pixels):
        pixels = normalization.normalize_range(pixels)
        try:
            clustering = clone(self.clustering).fit(pixels)
            colors = np.array([np.median(pixels[clustering.labels_ == label], axis=0)
                               for label in self.ordering(clustering.labels_, pixels)])
            return normalization.denormalize_range(colors)
        except ValueError:
            return np.empty((0, 3), dtype=np.uint8)


class MeanClusteringColorExtractor(ClusteringColorExtractor):
    """Color extractor based on clustering algorithm with mean of cluster colors"""

    def __init__(self, clustering):
        """
        Args:
            clustering: sklearn clustering model with fit() method and labels_ attribute
        """
        super().__init__(clustering)

    def extract_from_pixels(self, pixels):
        cluster_colors = super().extract_from_pixels(pixels)
        return np.atleast_2d(cluster_colors.mean(axis=0).astype(np.uint8))
