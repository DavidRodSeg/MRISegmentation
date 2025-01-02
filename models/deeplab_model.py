"""
Pretrained DeepLabV3 model adapted for binary segmentation with 4-channel input images.
"""

import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, deeplabv3


class Pretrained_Model(nn.Module):
    """
    Defines the fine-tuned model of the DeepLabV3 pretrained model.

    Args:
        in_ch (int): Number of input channels.
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.convolution = nn.Conv2d(in_ch, 3, (3,3), padding="same")
        self.pretrained_model = deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.DEFAULT)
        self.pretrained_model.classifier = deeplabv3.DeepLabHead(2048, 1)
        self.pretrained_model.aux_classifier = None
        self.sigmoid = nn.Sigmoid()

        for name, param in self.pretrained_model.backbone.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False  

    def forward(self, x):
        x = self.convolution(x)
        x = self.pretrained_model(x)["out"] # Shape: (batch_size, num_classes, height, width)
        output = self.sigmoid(x)

        return output