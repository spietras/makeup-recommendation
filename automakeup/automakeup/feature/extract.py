import numpy as np

from automakeup.feature.face import ClusteringIrisShapeExtractor
from automakeup.feature.makeup import LipstickColorExtractor, EyeshadowColorExtractor
from imagine.color.extract import MedianColorExtractor
from imagine.functional.functional import ImageOperation, Batchable
from imagine.shape import operations
from imagine.shape.segment import ParsingSegmenter


class ColorsFeatureExtractor(Batchable, ImageOperation):
    def __init__(self,
                 parser,
                 color_extractor=MedianColorExtractor(),
                 iris_extractor=ClusteringIrisShapeExtractor()):
        super().__init__()
        self.out_codes = {"skin": 1,
                          "hair": 2,
                          "lips": 3,
                          "eyes": 4}
        self.segmenter = ParsingSegmenter(parser, parts_map={"skin": self.out_codes["skin"],
                                                             "hair": self.out_codes["hair"],
                                                             "u_lip": self.out_codes["lips"],
                                                             "l_lip": self.out_codes["lips"],
                                                             "l_eye": self.out_codes["eyes"],
                                                             "r_eye": self.out_codes["eyes"]})
        self.extractor = color_extractor
        self.iris_extractor = iris_extractor

    def perform(self, faces, **kwargs):
        segmented = self.segmenter(faces)

        return self.stack([self._extract_single(f, s) for f, s in zip(faces, segmented)])

    def _extract_single(self, img, segmented):
        return np.concatenate([
            self._skin(img, segmented),
            self._hair(img, segmented),
            self._lips(img, segmented),
            self._eyes(img, segmented)
        ])

    def _simple_extract(self, img, segmented, part):
        colors = self.extractor.extract(img, segmented == self.out_codes[part])
        if colors is None:
            return np.zeros(3, dtype=np.uint8)
        return colors[0]

    def _skin(self, img, segmented):
        return self._simple_extract(img, segmented, "skin")

    def _hair(self, img, segmented):
        return self._simple_extract(img, segmented, "hair")

    def _lips(self, img, segmented):
        return self._simple_extract(img, segmented, "lips")

    def _eyes(self, img, segmented):
        eyes_mask = segmented == self.out_codes["eyes"]
        img_cropped, eye_mask_cropped = self._crop_to_biggest_eye(img, eyes_mask)
        iris_mask = self.iris_extractor.extract(img_cropped, eye_mask_cropped)
        colors = self.extractor.extract(img_cropped, iris_mask)
        if colors is None:
            return np.zeros(3, dtype=np.uint8)
        return colors[0]

    @staticmethod
    def _crop_to_biggest_eye(img, eyes_mask):
        # find biggest contour from mask and crop to it
        biggest_eye_contour = operations.biggest_contour(eyes_mask)
        eye_rect = operations.bounding_rect(biggest_eye_contour)
        if eye_rect is None:
            return img, eyes_mask
        eye_rect_square = operations.safe_rect(operations.squarisize(eye_rect), img.shape, allow_scaling=True)
        crop = operations.Crop(eye_rect_square)
        img_cropped = crop(img)
        eye_mask_cropped = crop(np.array(eyes_mask, dtype=np.uint8))
        # add erosion to get rid of uncertain edge
        eye_mask_cropped = operations.Erode(round(0.1 * eye_rect.height()))(eye_mask_cropped)
        return img_cropped, eye_mask_cropped != 0


class FacenetFeatureExtractor(Batchable, ImageOperation):
    def __init__(self, facenet):
        super().__init__()
        self.facenet = facenet

    def perform(self, faces, **kwargs):
        return self.facenet.embed(faces)


class MakeupExtractor(Batchable, ImageOperation):
    def __init__(self,
                 parser,
                 lipstick_extractor=LipstickColorExtractor(),
                 eyeshadow_extractor=EyeshadowColorExtractor()):
        super().__init__()
        self.out_codes = {"skin": 1,
                          "eyes": 2,
                          "lips": 4}
        self.segmenter = ParsingSegmenter(parser, parts_map={"skin": self.out_codes["skin"],
                                                             "l_eye": self.out_codes["eyes"],
                                                             "r_eye": self.out_codes["eyes"],
                                                             "u_lip": self.out_codes["lips"],
                                                             "l_lip": self.out_codes["lips"]})
        self.lipstick_extractor = lipstick_extractor
        self.eyeshadow_extractor = eyeshadow_extractor

    def perform(self, faces, **kwargs):
        segmented = self.segmenter(faces)

        return self.stack([self._extract_single(f, s) for f, s in zip(faces, segmented)])

    def _extract_single(self, img, segmented):
        return np.concatenate([
            self.lipstick_extractor.extract(img, segmented == self.out_codes["lips"]).flatten(),
            self.eyeshadow_extractor.extract(img,
                                             segmented == self.out_codes["skin"],
                                             segmented == self.out_codes["eyes"]).flatten()
        ])
