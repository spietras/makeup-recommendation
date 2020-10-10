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

        self.device = device

        pnet = models.PNet()
        with pkg_resources.path("{}.resources".format(__package__), pnet_model_path) as p:
            pnet.load_state_dict(torch.load(p, map_location=self.device))
        pnet.to(device)
        pnet.eval()

        rnet = models.RNet()
        with pkg_resources.path("{}.resources".format(__package__), rnet_model_path) as p:
            rnet.load_state_dict(torch.load(p, map_location=self.device))
        rnet.to(device)
        rnet.eval()

        onet = models.ONet()
        with pkg_resources.path("{}.resources".format(__package__), onet_model_path) as p:
            onet.load_state_dict(torch.load(p, map_location=self.device))
        onet.to(device)
        onet.eval()

        self.mtcnn = models.MTCNN(pnet,
                                  rnet,
                                  onet,
                                  device,
                                  min_face_size=min_face_size)
        self.mtcnn.to(device)
        self.mtcnn.eval()

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
            bbs, probs = self.mtcnn(imgs)

        return [bb.astype(np.integer) if bb is not None else None for bb in bbs], probs
