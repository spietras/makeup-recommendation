import importlib.resources as pkg_resources

import torch

from facenet.models import InceptionResnetV1

facenet_model_path = "inception_resnet_v1_vggface2.pt"


class Facenet:
    """Get 512-dimensional embeddings of faces from face images"""

    def __init__(self, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.net = InceptionResnetV1()

        with pkg_resources.path("{}.resources".format(__package__), facenet_model_path) as p:
            self.net.load_state_dict(torch.load(p))

        self.net.to(self.device)
        self.net.eval()

    @staticmethod
    def _normalize(imgs):
        return torch.tensor(imgs, dtype=torch.float).sub(127.5).div(128.).permute(0, 3, 1, 2)

    def embed(self, imgs):
        """
        Arguments:
            imgs - numpy array of shape (N, height, width, channels) with image data in RGB in uint8 big enough for
                   convolutions. model was trained on 160x160 images.

        Returns:
            numpy array of shape (N, 512) with embeddings
        """
        with torch.no_grad():
            imgs = self._normalize(imgs).to(self.device)
            embedded = self.net(imgs).cpu().numpy()

        return embedded
