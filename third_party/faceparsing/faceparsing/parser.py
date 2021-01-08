import importlib.resources as pkg_resources

import torch

from faceparsing.networks import BiSeNet

model_file = "bisenet.pth"


class FaceParser:
    codes = {
        0: 'bg',
        1: 'skin',
        2: 'l_brow',
        3: 'r_brow',
        4: 'l_eye',
        5: 'r_eye',
        6: 'eye_g',
        7: 'l_ear',
        8: 'r_ear',
        9: 'ear_r',
        10: 'nose',
        11: 'mouth',
        12: 'u_lip',
        13: 'l_lip',
        14: 'neck',
        15: 'neck_l',
        16: 'cloth',
        17: 'hair',
        18: 'hat'
    }

    def __init__(self, device=torch.device('cpu')):
        self.device = device

        with pkg_resources.path("{}.resources".format(__package__), model_file) as p:
            self.net = BiSeNet.load(torch.load(p, map_location=self.device))

        self.net.to(self.device)
        self.net.eval()

    @staticmethod
    def _normalize(imgs):
        mean = torch.as_tensor((0.485, 0.456, 0.406), dtype=torch.float)
        std = torch.as_tensor((0.229, 0.224, 0.225), dtype=torch.float)
        return torch.as_tensor(imgs, dtype=torch.float).div(255.).sub(mean).div(std).permute(0, 3, 1, 2)

    def parse(self, imgs):
        """
        Parse images into parts

        Args:
            imgs - numpy array of shape (N, height, width, 3) in RGB with values in [0-255]

        Returns:
            numpy array of shape (N, height, width) with values of codes in pixels recognized as parts
        """
        with torch.no_grad():
            imgs = self._normalize(imgs).to(self.device)
            out = self.net(imgs)[0]
            parsing = out.cpu().numpy().argmax(1)

        return parsing
