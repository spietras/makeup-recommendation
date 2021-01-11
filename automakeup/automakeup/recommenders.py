from abc import ABC, abstractmethod
import numpy as np


class Recommender(ABC):
    @abstractmethod
    def keys(self):
        return NotImplemented

    @abstractmethod
    def values(self, *args):
        return NotImplemented

    def recommend(self, *args):
        keys = self.keys()
        values = self.values(*args)
        return {key: value for key, value in zip(keys, values)}


class MakeupRecommender(Recommender, ABC):
    def keys(self):
        return ["lipstick_color", "eyeshadow_outer_color", "eyeshadow_middle_color", "eyeshadow_inner_color", "skin", "hair", "lips", "eyes"]


class DummyRecommender(MakeupRecommender):
    def values(self):
        return [[255, 0, 0],
                [210, 105, 30],
                [210, 105, 30],
                [210, 105, 30]]


class EncodingRecommender(MakeupRecommender):
    def __init__(self, bb_finder, face_extractor, feature_extractor, encoded_recommender):
        self.bb_finder = bb_finder
        self.face_extractor = face_extractor
        self.feature_extractor = feature_extractor
        self.encoded_recommender = encoded_recommender

    def values(self, image):
        bb = self.bb_finder.find(image)
        face = self.face_extractor.extract(image, bb)
        features = self.feature_extractor(face)
        y = self.encoded_recommender.recommend(features)
        y = np.append(y, features)
        return [y[i:i + 3].tolist() for i in range(0, len(y), 3)]
