import importlib.resources as pkg_resources

import numpy as np
import torch

from mtcnn import networks

model_file = "mtcnn.pt"


class MTCNN:
    """Find bounding boxes of faces in image using MTCNN"""

    def __init__(self, min_face_size=20, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        with pkg_resources.path("{}.resources".format(__package__), model_file) as p:
            self.net = networks.MTCNN.load(torch.load(p, map_location=self.device), min_face_size=min_face_size)

        self.net.to(device)
        self.net.eval()

    def find(self, imgs):
        """
        Arguments:
            imgs - numpy array of shape (N, height, width, channels) with image data in RGB in uint8

        Returns:
            tuple with list of length N of numpy array of shape (faces, 4) with bounding boxes and list with length N
            of detection probabilities. Returned boxes will be sorted in descending order by detection probability
        """
        with torch.no_grad():
            imgs = torch.tensor(imgs).permute(0, 3, 1, 2).to(self.device)
            bbs, probs = self.net(imgs)

        return [bb.astype(np.integer) if bb is not None else None for bb in bbs], probs
