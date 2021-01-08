from abc import ABC, abstractmethod

import numpy as np

from imagine.color import conversion
from imagine.functional import functional as f
from imagine.functional.functional import ImageBatchClassifier
from imagine.helpers import normalization


class EncodedRecommender(ABC):
    @abstractmethod
    def recommend(self, features):
        return NotImplemented


class GanetteRecommender(EncodedRecommender):
    def __init__(self, model, x_scaler, y_scaler):
        super().__init__()
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.preprocess = f.Join([
            f.Rearrange("(f fs) -> 1 f fs", fs=3),
            normalization.ToUInt8(),
            conversion.RgbToLab,
            f.Rearrange("1 f fs -> 1 (f fs)", fs=3)
        ])
        self.postprocess = f.Join([
            f.Rearrange("1 (f fs) -> 1 f fs", fs=3),
            f.Lambda(lambda x: x.astype(np.uint8), ImageBatchClassifier()),
            conversion.LabToRgb,
            f.Rearrange("1 f fs -> (f fs)", fs=3)
        ])

    def recommend(self, features):
        y = self.y_scaler.transform(self.preprocess(features))
        x = self.model.sample(y)
        return self.postprocess(self.x_scaler.inverse_transform(x))
