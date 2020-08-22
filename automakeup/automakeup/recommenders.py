from abc import ABC, abstractmethod


class Recommender(ABC):
    @abstractmethod
    def recommend(self, input, *args):
        return NotImplemented


class DummyRecommender(Recommender):
    def recommend(self, input, *args):
        return {
            "lips_color": "red"
        }


class EncodingRecommender(Recommender):
    def __init__(self, face_extractor, feature_extractor, encoded_recommender):
        self.face_extractor = face_extractor
        self.feature_extractor = feature_extractor
        self.encoded_recommender = encoded_recommender

    def recommend(self, input, *args):
        face = self.face_extractor.extract(input)
        features = self.feature_extractor(face)
        y = self.encoded_recommender.recommend(features)
        return {
            "k1": y[0],
            "k2": y[1]
        }
