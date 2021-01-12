from abc import ABC, abstractmethod

import numpy as np


class Results:
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)


class Recommender(ABC):
    @abstractmethod
    def recommend(self, *args):
        return NotImplemented


class MakeupRecommender(Recommender, ABC):
    class MakeupResults(Results):
        def __init__(self, skin_color, hair_color, lips_color, eyes_color, lipstick_color,
                     eyeshadow_outer_color, eyeshadow_middle_color, eyeshadow_inner_color):
            super().__init__(skin_color=skin_color, hair_color=hair_color, lips_color=lips_color,
                             eyes_color=eyes_color, lipstick_color=lipstick_color,
                             eyeshadow_outer_color=eyeshadow_outer_color,
                             eyeshadow_middle_color=eyeshadow_middle_color,
                             eyeshadow_inner_color=eyeshadow_inner_color)


class DummyRecommender(MakeupRecommender):
    def recommend(self):
        return self.MakeupResults(*np.random.randint(0, 256, (8, 3)).tolist())


class EncodingRecommender(MakeupRecommender):
    def __init__(self, bb_finder, face_extractor, feature_extractor, encoded_recommender):
        self.bb_finder = bb_finder
        self.face_extractor = face_extractor
        self.feature_extractor = feature_extractor
        self.encoded_recommender = encoded_recommender

    def recommend(self, image):
        bb = self.bb_finder.find(image)
        face = self.face_extractor.extract(image, bb)
        features = self.feature_extractor(face)
        y = self.encoded_recommender.recommend(features)
        out = np.append(features, y)
        return self.MakeupResults(*[out[i:i + 3].tolist() for i in range(0, len(out), 3)])
