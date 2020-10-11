from abc import ABC, abstractmethod

import cv2
import numpy as np
from sklearn.cluster import KMeans

from automakeup.feature.utils import first_channel_ordering
from imagine.color import conversion
from imagine.shape import operations
from imagine.shape.segment import ClusteringSegmenter


class IrisShapeExtractor(ABC):
    @abstractmethod
    def extract(self, img, eye_mask):
        return NotImplemented


class ThresholdingIrisShapeExtractor(IrisShapeExtractor):
    def __init__(self,
                 lower_quantile=0.1,
                 upper_quantile=0.5):
        super().__init__()
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def extract(self, img, eye_mask):
        img = conversion.RgbToLab(img)
        l_channel = img[..., 0]
        lower_quantile_threshold = np.quantile(l_channel, self.lower_quantile)
        upper_quantile_threshold = np.quantile(l_channel, self.upper_quantile)
        return (img[..., 0] >= lower_quantile_threshold) & (img[..., 0] <= upper_quantile_threshold) & eye_mask


class ClusteringIrisShapeExtractor(IrisShapeExtractor):
    def __init__(self, clustering_config=(KMeans(n_clusters=6),
                                          first_channel_ordering,
                                          [1, 2, 3])):
        super().__init__()
        clustering, ordering, chosen_clusters = clustering_config
        self.segmenter = ClusteringSegmenter(clustering, ordering, parts_map={p: p for p in chosen_clusters})

    def extract(self, img, eye_mask):
        img = conversion.RgbToLab(img)
        return self.segmenter(img, masks=eye_mask) != 0


class HoughCircleIrisShapeExtractor(IrisShapeExtractor):
    def __init__(self, method=cv2.HOUGH_GRADIENT_ALT, dp=1.25, min_dist=100, param1=1, param2=0.0, pupil_ratio=0.2):
        super().__init__()
        self.method = method
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.pupil_ratio = pupil_ratio

    def extract(self, img, eye_mask):
        circles = cv2.HoughCircles(conversion.RgbToGray(img), self.method, self.dp, self.min_dist,
                                   param1=self.param1, param2=self.param2, maxRadius=int(0.5 * img.shape[0]))
        if circles is None:
            return np.zeros(img.shape[:2], dtype=bool)

        x, y, r = np.around(circles[0, 0, :]).astype(np.int)

        circle_mask = operations.circle_mask(img.shape, (x, y), r)
        pupil_mask = operations.circle_mask(img.shape, (x, y), int(self.pupil_ratio * r))
        return eye_mask & circle_mask & ~pupil_mask
