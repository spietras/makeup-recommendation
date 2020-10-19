import importlib.resources as pkg_resources
import logging

import configargparse
import dlib
import torch

from automakeup import dlib_predictor_path
from automakeup.face import extract as face_extraction
from automakeup.face.bounding import MTCNNBoundingBoxFinder
from automakeup.feature import extract as feature_extraction
from data import IndexedImageDictDataLoader, MakeupDataset, DataFrameCsvSaver
from facenet import Facenet
from faceparsing import FaceParser
from mtcnn import MTCNN
from pipeline import PreprocessingPipeline
from preprocessors import MakeupDataPreprocessor


def parse_args():
    with pkg_resources.path("resources", "config.yaml") as config_path:
        argparser = configargparse.ArgParser(prog=__package__,
                                             description="{} - makeup data preprocessing".format(__package__),
                                             default_config_files=[str(config_path)])
        argparser.add_argument("directory",
                               help="path to directory containing makeup images")
        argparser.add_argument("output_directory",
                               help="path to directory that should contain output data file")
        argparser.add_argument('--config', is_config_file=True,
                               help='config file path')
        argparser.add_argument("--output_file", default="data",
                               help="name of the output data file")
        argparser.add_argument("--batchsize", type=int, default=1,
                               help="batch size (how many images to process at once)")
        argparser.add_argument("--facesize", type=int, default=512,
                               help="edge size of square that faces will be resized to")
        argparser.add_argument("--limit", type=int, default=10,
                               help="after how many images should the output be written to disk")
        argparser.add_argument("--method", choices=["colors", "facenet"], default="colors",
                               help="method of feature encoding")
        args = argparser.parse_args()
    return args


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def config_logging():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')


def get_method_config(device, directory, batchsize, face_extractor, face_feature_extractor, makeup_feature_extractor):
    bb_finder = MTCNNBoundingBoxFinder(MTCNN(device=device))
    data_loader = IndexedImageDictDataLoader(MakeupDataset(directory),
                                             batch_size=batchsize,
                                             align=face_extraction.ExtractFace(bb_finder, face_extractor))
    preprocessor = MakeupDataPreprocessor(face_feature_extractor, makeup_feature_extractor)
    return data_loader, preprocessor


def get_colors_config(device, facesize, directory, batchsize):
    face_extractor = face_extraction.SimpleFaceExtractor(output_size=facesize, bb_scale=1.5)
    parser = FaceParser(device=device)
    face_feature_extractor = feature_extraction.ColorsFeatureExtractor(parser)
    makeup_feature_extractor = feature_extraction.MakeupExtractor(parser)
    return get_method_config(device, directory, batchsize, face_extractor, face_feature_extractor,
                             makeup_feature_extractor)


def get_facenet_config(device, facesize, directory, batchsize):
    with dlib_predictor_path() as p:
        predictor = dlib.shape_predictor(str(p))
    face_extractor = face_extraction.AligningDlibFaceExtractor(output_size=facesize, predictor=predictor)
    parser = FaceParser(device=device)
    face_feature_extractor = feature_extraction.FacenetFeatureExtractor(Facenet(device=device))
    makeup_feature_extractor = feature_extraction.MakeupExtractor(parser)
    return get_method_config(device, directory, batchsize, face_extractor, face_feature_extractor,
                             makeup_feature_extractor)


if __name__ == '__main__':
    args = parse_args()
    device = get_device()
    config_logging()

    logger = logging.getLogger("main")
    logger.info("Using device = {}".format(str(device)))
    logger.info("Loading")

    config_function = get_colors_config if args.method == "colors" else get_facenet_config
    data_loader, preprocessor = config_function(device, args.facesize, args.directory, args.batchsize)

    data_saver = DataFrameCsvSaver(args.output_directory, args.output_file, limit=args.limit)

    logger.info("Loaded")

    pipeline = PreprocessingPipeline(data_loader, preprocessor.preprocess, data_saver)
    pipeline.run()
