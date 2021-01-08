import pickle
from abc import ABC, abstractmethod

import torch

import automakeup
from automakeup.encoded_recommendation import GanetteRecommender
from automakeup.face.bounding import MTCNNBoundingBoxFinder
from automakeup.face.extract import SimpleFaceExtractor
from automakeup.feature.extract import ColorsFeatureExtractor
from automakeup.recommenders import EncodingRecommender
from faceparsing import FaceParser
from ganette import Ganette
from mtcnn import MTCNN


class Pipeline(ABC):
    @abstractmethod
    def run(self, *args):
        return NotImplemented


class GanettePipeline(Pipeline):
    def __init__(self,
                 device=torch.device('cpu'),
                 face_size=512,
                 bb_scale=1.5):
        super().__init__()
        bb_finder = self._get_bb_finder(device)
        face_extractor = self._get_face_extractor(face_size, bb_scale)
        feature_extractor = self._get_feature_extractor(device)
        self.recommender = self._get_recommender(bb_finder, face_extractor, feature_extractor, device)

    @staticmethod
    def _get_bb_finder(device):
        return MTCNNBoundingBoxFinder(MTCNN(device=device))

    @staticmethod
    def _get_face_extractor(face_size, bb_scale=1.5):
        return SimpleFaceExtractor(output_size=face_size, bb_scale=bb_scale)

    @staticmethod
    def _get_feature_extractor(device):
        parser = FaceParser(device=device)
        return ColorsFeatureExtractor(parser)

    @staticmethod
    def _get_recommender(bb_finder, face_extractor, feature_extractor, device):
        with automakeup.ganette_model_path() as p:
            with open(p, "rb") as f:
                model = Ganette.unpickle(f).to(device)
        with automakeup.ganette_x_scaler_path() as p:
            with open(p, "rb") as f:
                x_scaler = pickle.load(f)
        with automakeup.ganette_y_scaler_path() as p:
            with open(p, "rb") as f:
                y_scaler = pickle.load(f)
        encoded_recommender = GanetteRecommender(model, x_scaler, y_scaler)
        return EncodingRecommender(bb_finder, face_extractor, feature_extractor, encoded_recommender)

    def run(self, img):
        return self.recommender.recommend(img)
