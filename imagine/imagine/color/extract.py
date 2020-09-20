from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import clone

from imagine import helpers
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
        img = helpers.normalize_images(img)
        img = conversion.RgbToLab.convert(img)
        return self.extract_normalized(img, mask)

    @abstractmethod
    def extract_normalized(self, img, mask):
        return NotImplemented


class PositionAgnosticExtractor(ColorExtractor, ABC):

    def extract_normalized(self, img, mask):
        pixels = img[mask == 1]
        if pixels.size == 0:
            return None
        return conversion.LabToRgb.convert(self.extract_from_pixels(pixels))

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

    def extract_from_pixels(self, pixels):
        return np.atleast_2d(pixels.mean(axis=0).astype(np.uint8))


class MedianColorExtractor(PositionAgnosticExtractor):

    def extract_from_pixels(self, pixels):
        return np.atleast_2d(np.median(pixels, axis=0).astype(np.uint8))


class ClusteringColorExtractor(PositionAgnosticExtractor):

    def __init__(self, clustering):
        super().__init__()
        self.clustering = clustering

    def extract_from_pixels(self, pixels):
        pixels = helpers.normalize_range(pixels)
        clustering = clone(self.clustering).fit(pixels)
        colors = np.array([pixels[clustering.labels_ == label].mean(axis=0)
                           for label in range(clustering.labels_.max() + 1)])
        return helpers.denormalize_range(colors)


class MeanClusteringColorExtractor(ClusteringColorExtractor):

    def __init__(self, clustering):
        super().__init__(clustering)

    def extract_from_pixels(self, pixels):
        cluster_colors = super().extract_from_pixels(pixels)
        return np.atleast_2d(cluster_colors.mean(axis=0).astype(np.uint8))
