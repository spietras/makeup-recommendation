from torch import nn
from torch.nn import functional as F

from facenet import layers
from modelutils import LoadableModule


class InceptionResnetV1(LoadableModule):
    """
    Inception Resnet V1 model

    Args:
        dropout_prob - dropout probability (between 0.0 and 1.0)
    """

    def __init__(self, dropout_prob=0.6):
        super().__init__()

        # Define layers
        self.conv2d_1a = layers.BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = layers.BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = layers.BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = layers.BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = layers.BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = layers.BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            layers.Block35(scale=0.17),
            layers.Block35(scale=0.17),
            layers.Block35(scale=0.17),
            layers.Block35(scale=0.17),
            layers.Block35(scale=0.17),
        )
        self.mixed_6a = layers.Mixed_6a()
        self.repeat_2 = nn.Sequential(
            layers.Block17(scale=0.10),
            layers.Block17(scale=0.10),
            layers.Block17(scale=0.10),
            layers.Block17(scale=0.10),
            layers.Block17(scale=0.10),
            layers.Block17(scale=0.10),
            layers.Block17(scale=0.10),
            layers.Block17(scale=0.10),
            layers.Block17(scale=0.10),
            layers.Block17(scale=0.10),
        )
        self.mixed_7a = layers.Mixed_7a()
        self.repeat_3 = nn.Sequential(
            layers.Block8(scale=0.20),
            layers.Block8(scale=0.20),
            layers.Block8(scale=0.20),
            layers.Block8(scale=0.20),
            layers.Block8(scale=0.20),
        )
        self.block8 = layers.Block8(no_relu=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        """
        Calculate embeddings given a batch of input image tensors.

        Arguments:
            x  - torch.tensor for shape (N, channels, height, width) in RGB in float in [-1.0, 1.0]
                 big enough for convolutions

        Returns:
            torch.tensor of shape (N, 512) with embeddings
        """

        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)

        x = F.normalize(x, p=2, dim=1)
        return x
