import importlib.resources as pkg_resources

import numpy as np
import torch

from mtcnn import models

pnet_model_path = "pnet.pt"
rnet_model_path = "rnet.pt"
onet_model_path = "onet.pt"


class MTCNN:
    """Find bounding boxes of faces in image using MTCNN"""

    def __init__(self, min_face_size=20, device=torch.device('cpu')):
        super().__init__()

        pnet = models.PNet()
        with pkg_resources.path("{}.resources".format(__package__), pnet_model_path) as p:
            pnet.load_state_dict(torch.load(p))

        rnet = models.RNet()
        with pkg_resources.path("{}.resources".format(__package__), rnet_model_path) as p:
            rnet.load_state_dict(torch.load(p))

        onet = models.ONet()
        with pkg_resources.path("{}.resources".format(__package__), onet_model_path) as p:
            onet.load_state_dict(torch.load(p))

        self.mtcnn = models.MTCNN(pnet,
                                  rnet,
                                  onet,
                                  min_face_size=min_face_size,
                                  device=device)

    def find(self, img):
        """
        Arguments:
            img - numpy array of shape (N, height, width, channels) with image data in RGB in uint8

        Returns:
            tuple with numpy array of shape (N, 4) with bounding boxes and list with length N of detection probabilities.
            Returned boxes will be sorted in descending order by detection probability
        """
        bbs, probs = self.mtcnn(img)
        return [bb.astype(np.integer) for bb in bbs], probs
