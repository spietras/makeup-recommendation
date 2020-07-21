from abc import ABC, abstractmethod


class Recommender(ABC):
    @abstractmethod
    def recommend(self, input, *args):
        return NotImplemented


class DummyRecommender(Recommender):
    def recommend(self, input, *args):
        return {
            "lips_color" : "red"
        }


class EncodingRecommender(Recommender):
    def __init__(self, face_extractor, feature_extractor, encoded_recommender):
        self.face_extractor = face_extractor
        self.feature_extractor = feature_extractor
        self.encoded_recommender = encoded_recommender

    def recommend(self, input, *args):
        pass


