import logging
import os
import pathlib
from abc import ABC, abstractmethod

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from imagine.color import conversion
from imagine.functional import functional as f

logger = logging.getLogger(__name__)


# Datasets


class IndexedTreeDataset(Dataset, ABC):
    def __init__(self, root_directory):
        super().__init__()
        self.directories = [d.path for d in os.scandir(root_directory) if d.is_dir()]

    def __getitem__(self, index):
        directory = self.directories[index]
        last_dir = pathlib.PurePath(directory).name
        return last_dir, self.get_from_directory(directory)

    def __len__(self):
        return len(self.directories)

    @abstractmethod
    def get_from_directory(self, directory):
        return NotImplemented


class ItemGetter(ABC):
    @abstractmethod
    def get_single(self, path):
        return NotImplemented


class LabelDictIndexedTreeDataset(IndexedTreeDataset, ItemGetter, ABC):
    def __init__(self, root_directory, labels, format="jpg"):
        super().__init__(root_directory)
        self.labels = labels
        self.format = format

    def get_from_directory(self, directory):
        return {label: self.get_single("{}/{}.{}".format(directory, label, self.format)) for label in self.labels}


class ImageGetter(ItemGetter):
    def get_single(self, path):
        return conversion.BgrToRgb(cv2.imread(path))


class MakeupDataset(LabelDictIndexedTreeDataset, ImageGetter):
    def __init__(self, root_directory, before_label="before", after_label="after", format="jpg"):
        super().__init__(root_directory, [before_label, after_label], format)


# Data loaders


class IndexedImageDictDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 align=f.Identity,
                 shuffle=True):
        super().__init__(dataset, batch_size, shuffle, collate_fn=self.Collator(align).collate)

    class Collator:
        def __init__(self, align):
            super().__init__()
            self.align = align

        def collate(self, data):
            indices = [d[0] for d in data]
            dicts = [d[1] for d in data]
            labels = dicts[0].keys()
            return indices, {l: self.collate_images([d[l] for d in dicts]) for l in labels}

        def collate_images(self, images):
            return np.stack([self.align(i) for i in images])


# Data savers


class DataSaver(ABC):
    @abstractmethod
    def save(self, *args):
        return NotImplemented


class PartialDataSaver(DataSaver, ABC):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    @abstractmethod
    def mem_size(self):
        return NotImplemented

    def save(self, *args):
        self.save_in_memory(*args)
        if self.mem_size() >= self.limit:
            self.dump()
            logger.info("Dumped data")

    @abstractmethod
    def save_in_memory(self, *args):
        return NotImplemented

    @abstractmethod
    def dump(self):
        return NotImplemented


class DataFrameFileSaver(PartialDataSaver, ABC):
    def __init__(self, file, limit=100):
        super().__init__(limit)
        self.file = file
        self.buffer = []
        self.size = 0
        self.first_dump = True

    def mem_size(self):
        return self.size

    def save_in_memory(self, rows):
        self.buffer.append(rows)
        self.size += len(rows)

    def dump(self):
        self.export(pd.concat(self.buffer))
        self.first_dump = False
        self.buffer = []
        self.size = 0

    @abstractmethod
    def export(self, df):
        return NotImplemented


class DataFrameCsvSaver(DataFrameFileSaver):
    def export(self, df):
        df.to_csv(self.file, mode="a", header=self.first_dump, index=False)
