from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import KMeans

from imagine import helpers


class ColorExtractor(ABC):
    @abstractmethod
    def extract(self, img, mask):
        """
        Extract colors from pixels

        Args:
            img - numpy array of shape (height, width, 3) in RGB with values either in [0-255] or [0.0-1.0]
            mask - numpy array of shape (height, width) with ones or Trues in pixels to process

        Returns:
            numpy array of shape (N, 3) with N extracted colors in RGB in [0-255] or None if mask is empty
        """
        return NotImplemented


class PositionAgnosticExtractor(ColorExtractor, ABC):

    def extract(self, img, mask):
        img = helpers.normalize_photo(img)
        img = helpers.to_lab(img)
        pixels = img[mask == 1]
        if pixels.size == 0:
            return None
        return helpers.to_rgb(self.extract_from_pixels(pixels))

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


class KMeansColorExtractor(PositionAgnosticExtractor):

    def __init__(self, k=1, seed=None):
        super().__init__()
        self.k = k
        self.seed = seed

    def extract_from_pixels(self, pixels):
        pixels = helpers.normalize_range(pixels)
        kmeans = KMeans(n_clusters=self.k, random_state=self.seed).fit(pixels)
        return helpers.denormalize_range(kmeans.cluster_centers_)


class MeanKMeansColorExtractor(KMeansColorExtractor):

    def __init__(self, k=3, seed=None):
        super().__init__(k, seed)

    def extract_from_pixels(self, pixels):
        cluster_colors = super().extract_from_pixels(pixels)
        return np.atleast_2d(cluster_colors.mean(axis=0).astype(np.uint8))
