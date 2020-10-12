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
    def __init__(self,
                 lower_cluster_cut=0.1,
                 upper_cluster_cut=0.6,
                 eye_clustering=KMeans(n_clusters=11),
                 cluster_ordering=first_channel_ordering):
        super().__init__()
        self.segmenter = ClusteringSegmenter(eye_clustering, ordering=cluster_ordering, bg_code=-1)
        self.lower_cluster_cut = lower_cluster_cut
        self.upper_cluster_cut = upper_cluster_cut

    def extract(self, img, eye_mask):
        img = conversion.RgbToLab(img)
        clustered = self.segmenter(img, masks=eye_mask)
        return self._remove_non_iris(clustered)

    def _remove_non_iris(self, clustered):
        k = clustered.max() + 1
        if k <= 1:
            return np.zeros(clustered.shape, dtype=np.bool)
        lower_cluster_threshold = np.quantile(range(k), self.lower_cluster_cut)
        upper_cluster_threshold = np.quantile(range(k), self.upper_cluster_cut)
        return (clustered >= lower_cluster_threshold) & (clustered <= upper_cluster_threshold) & (clustered != -1)


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