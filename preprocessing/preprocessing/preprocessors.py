from abc import ABC, abstractmethod

import pandas as pd


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, data):
        return NotImplemented


class MakeupDataPreprocessor(Preprocessor):
    def __init__(self,
                 before_feature_extractor,
                 after_feature_extractor):
        super().__init__()
        self.before_feature_extractor = before_feature_extractor
        self.after_feature_extractor = after_feature_extractor

    def preprocess(self, data):
        id = data[0]
        before = data[1]["before"]
        after = data[1]["after"]

        indices = pd.DataFrame(id if isinstance(id, list) else [id], columns=["id"])
        before_features = pd.DataFrame(self.before_feature_extractor(before),
                                       columns=self.before_feature_extractor.labels())
        after_features = pd.DataFrame(self.after_feature_extractor(after),
                                      columns=self.after_feature_extractor.labels())

        return pd.concat([indices, before_features, after_features], axis=1)
