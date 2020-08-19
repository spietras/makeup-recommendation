from abc import ABC, abstractmethod


class EncodedRecommender(ABC):
    @abstractmethod
    def recommend(self, features):
        return NotImplemented
