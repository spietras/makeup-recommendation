from math import sqrt

import cv2
import numpy as np
from sklearn import clone
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

from automakeup.feature.utils import first_channel_ordering
from imagine.color import conversion
from imagine.color.extract import MedianColorExtractor, ClusteringColorExtractor
from imagine.shape import operations
from imagine.shape.segment import ClusteringSegmenter


class EyeshadowShapeExtractor:
    def __init__(self,
                 skin_color_extractor=MedianColorExtractor(),
                 eyeshadow_clustering=AgglomerativeClustering(6, linkage='average'),
                 skin_color_cluster_classifier=KNeighborsClassifier(n_neighbors=3),
                 outer_eye_factor=2.25,
                 inner_eye_factor=0.25):
        super().__init__()
        self.skin_color_extractor = skin_color_extractor
        self.eyeshadow_segmenter = ClusteringSegmenter(eyeshadow_clustering, bg_code=-1)
        self.skin_color_cluster_classifier = skin_color_cluster_classifier
        self.outer_eye_factor = outer_eye_factor
        self.inner_eye_factor = inner_eye_factor

    def extract(self, img, skin_mask, eyes_mask):
        around_eye_mask = self._area_around_eye(eyes_mask, skin_mask)
        skin_color = self._get_skin_color(img, skin_mask)
        clustered = self.eyeshadow_segmenter(conversion.RgbToLab(img), masks=around_eye_mask)
        return self._remove_bad_areas(img, clustered, skin_color)

    def _area_around_eye(self, eyes_mask, skin_mask):
        eye_contour, eye_mask = self._get_bigger_eye(eyes_mask)
        if eye_contour is None or eye_contour.size == 0:
            return np.zeros(eyes_mask.shape[:2], dtype=np.bool)
        eye_area = cv2.contourArea(eye_contour)
        eye = np.array(eye_mask, dtype=np.uint8)
        outer_kernel_size = max(int(sqrt(eye_area) * self.outer_eye_factor), 2)
        inner_kernel_size = max(int(sqrt(eye_area) * self.inner_eye_factor), 1)
        outer_dilated_eye = operations.Dilate((outer_kernel_size, outer_kernel_size))(eye)
        inner_dilated_eye = operations.Dilate((inner_kernel_size, inner_kernel_size))(eye)
        return (outer_dilated_eye > 0) & ~(inner_dilated_eye > 0) & skin_mask

    @staticmethod
    def _get_bigger_eye(eyes_mask):
        eye_contour = operations.biggest_contour(eyes_mask)
        eye_mask = operations.fill_contour(eye_contour, eyes_mask.shape[:2])
        return eye_contour, eye_mask

    def _get_skin_color(self, img, skin_mask):
        return self.skin_color_extractor.extract(img, skin_mask)[0]

    def _remove_bad_areas(self, img, clustered, skin_color):
        skin_color_lab = conversion.RgbToLab(np.array([[skin_color]])).reshape(1, 3)
        black_color_lab = conversion.RgbToLab(np.array([[[0, 0, 0]]], dtype=np.uint8)).reshape(1, 3)

        non_background = clustered != -1
        if not non_background.any():
            return non_background

        pixels = conversion.RgbToLab(img)[non_background]
        labels = clustered[non_background]

        neigh = clone(self.skin_color_cluster_classifier).fit(pixels, labels)
        skin_cluster, eyelashes_cluster = neigh.predict(np.concatenate([skin_color_lab, black_color_lab]))
        return (clustered != skin_cluster) & (clustered != eyelashes_cluster) & (clustered != -1)


class EyeshadowColorExtractor:
    def __init__(self,
                 shape_extractor=EyeshadowShapeExtractor(),
                 color_extractor=ClusteringColorExtractor(KMeans(3),
                                                          ordering=first_channel_ordering)):
        super().__init__()
        self.shape_extractor = shape_extractor
        self.color_extractor = color_extractor

    def extract(self, img, skin_mask, eyes_mask):
        eyeshadow_area = self.shape_extractor.extract(img, skin_mask, eyes_mask)
        return self.color_extractor.extract(img, eyeshadow_area)


class LipstickColorExtractor:
    def __init__(self, color_extractor=MedianColorExtractor()):
        super().__init__()
        self.color_extractor = color_extractor

    def extract(self, img, lips_mask):
        return self.color_extractor.extract(img, lips_mask)
