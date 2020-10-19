import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Pipeline(ABC):
    @abstractmethod
    def run(self):
        return NotImplemented


class PreprocessingPipeline(Pipeline):
    def __init__(self, data_loader, preprocessing, data_saver):
        self.data_loader = data_loader
        self.preprocessing = preprocessing
        self.data_saver = data_saver

    def run(self):
        logger.info("Pipeline start")
        with self.data_saver as s:
            total_batches = len(self.data_loader)
            for i, batch in enumerate(self.data_loader):
                logger.info("Loaded batch {}/{}".format(i, total_batches))
                preprocessed = self.preprocessing(batch)
                logger.info("Preprocessed batch {}/{}, shape = {}".format(i, total_batches, preprocessed.shape))
                s.save(preprocessed)
                logger.info("Saved batch {}/{}".format(i, total_batches))
        logger.info("Pipeline end")
