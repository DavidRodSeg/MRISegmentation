"""
Definition of double convolution and upsampling and downsumpling blocks based on
U-Net: Convolutional Networks for Biomedical Image Segmentation (https://arxiv.org/pdf/1505.04597).
"""
import torch
from torch import nn

class DoubleConv2D(nn.Module):
    """
    Class for the application of two 2D convolutions, each followed
    by a batch normalization layer and a ReLU layer (in that order).

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    Returns:
        DoubleConv2D (object): Instance of the DoubleConv2D class.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv_2d = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.double_conv_2d(x)



# Original UNet. Build as specified in U-Net: Convolutional Networks for Biomedical Image Segmentation.
# Input and output sizes do not match due to the unpadded convolutions, so for using it the target masks
# must be cropped.
class DownBlockOriginal(nn.Module):
    """
    Class for the construction of a downsampling block that
    returns the skip connection and the downsampled image.    

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    Returns:
        DownBlock (object): Instance of the DownBlock class.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv_2d = DoubleConv2D(in_ch, out_ch)
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.double_conv_2d(x)
        x = self.pooling(skip)     
        return (x, skip)
    

class UpBlockOriginal(nn.Module):
    """
    Class for the construction of a upsampling block that
    returns the skip connection and the upsampled image.    

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    Returns:
        UpBlock (object): Instance of the UpBlock class.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convtranspose = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.double_conv_2d = DoubleConv2D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.convtranspose(x)
        x = torch.cat((x, skip), dim=1)
        x = self.double_conv_2d(x)
        return x
    

class OriginalUNet(nn.Module):
    def __init__(self, in_ch = 1, out_classes = 1):
        super().__init__()
        # Encoder
        self.down1 = DownBlockOriginal(in_ch, 64)
        self.down2 = DownBlockOriginal(64, 128)
        self.down3 = DownBlockOriginal(128, 256)
        self.down4 = DownBlockOriginal(256, 512)
        # Bottleneck
        self.conv = DoubleConv2D(512, 1024)
        # Decoder
        self.up1 = UpBlockOriginal(1024, 512)
        self.up2 = UpBlockOriginal(512, 256)
        self.up3 = UpBlockOriginal(256, 128)
        self.up4 = UpBlockOriginal(128, 64)
        # Final convolution
        self.last_conv = nn.Sequential(
            nn.Conv2d(64, out_classes, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x = self.conv(x)
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        output = self.last_conv(x)

        return output
    


# Modified version of the UNet that uses padded convolutions so the target masks
# can have the same size as the input images. The padding of the convolutions is
# 1 or "same" so the output has the same size as the input.
class DownBlock(nn.Module):
    """
    Class for the construction of a downsampling block that
    returns the skip connection and the downsampled image.    

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    Returns:
        DownBlock (object): Instance of the DownBlock class.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv_2d = DoubleConv2D(in_ch, out_ch)
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.double_conv_2d(x)
        x = self.pooling(skip)     
        return (x, skip)


class UpBlock(nn.Module):
    """
    Class for the construction of a upsampling block that
    returns the skip connection and the upsampled image.    

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    Returns:
        UpBlock (object): Instance of the UpBlock class.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convtranspose = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.double_conv_2d = DoubleConv2D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.convtranspose(x)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv_2d(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch = 1, out_classes = 1):
        super().__init__()
        # Encoder
        self.down1 = DownBlock(in_ch, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        # Bottleneck
        self.conv = DoubleConv2D(512, 1024)
        # Decoder
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        # Final convolution
        self.last_conv = nn.Sequential(
            nn.Conv2d(64, out_classes, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x = self.conv(x)
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        output = self.last_conv(x)

        return output