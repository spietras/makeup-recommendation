import importlib.resources as pkg_resources

import torch
import torchvision.transforms as transforms

from faceparsing.model import BiSeNet

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
        self.n_classes = 19
        self.device = device

        self.net = BiSeNet(n_classes=self.n_classes)
        self.net.to(self.device)

        with pkg_resources.path("{}.resources".format(__package__), model_file) as p:
            self.net.load_state_dict(torch.load(p))

        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def parse(self, img):
        with torch.no_grad():
            img = self.to_tensor(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        return parsing
