"""
UNet models and variations. Defintion of the fundamental blocks and the models itself.
"""

import torch
from torch import nn



# Original UNet. Build as specified in U-Net: Convolutional Networks for Biomedical Image Segmentation
# (https://arxiv.org/pdf/1505.04597). Input and output sizes do not match due to the unpadded convolutions, 
# so for using it the target masks must be cropped.
class DoubleConv2DOriginal(nn.Module):
    """
    Class for the application of two 2D convolutions without batch normalization,
    each followed by a batch normalization layer and a ReLU layer (in that order).

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
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding="same"),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.double_conv_2d(x)
    

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
        self.double_conv_2d = DoubleConv2DOriginal(in_ch, out_ch)
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
        self.conv_transpose = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.double_conv_2d = DoubleConv2DOriginal(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.conv_transpose(x)
        x = torch.cat((x, skip), dim=1)
        x = self.double_conv_2d(x)
        return x
    

class OriginalUNet(nn.Module):
    """
    UNet model for image segmentation as described in the original article. 
    The output image dimensions are smaller than the input image dimensions.

    Args:
        in_ch (int, optional): Number of input channels (default: 1).
        out_classes (int, optional): Number of output classes. Use 1 for binary segmentation 
            (default: 1).
    """
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
        self.conv_transpose = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.double_conv_2d = DoubleConv2D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.conv_transpose(x)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv_2d(x)
        return x


class UNet(nn.Module):
    """
    UNet model for image segmentation adapted for the output to be of the
    same size as the input.

    Args:
        in_ch (int, optional): Number of input channels (default: 1).
        out_classes (int, optional): Number of output classes. Use 1 for binary segmentation 
            (default: 1).
    """
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



# UNet with residual connections implementations. The residual connections
# helps the network to avoid vanishing and exploding gradient problems
# during the training of deep NN. Based on the paper Road Extraction by 
# Deep Residual U-Net (https://arxiv.org/pdf/1711.10684)
class ResidualBlock(nn.Module): # PROBAR A QUITAR ESTO Y AÑADIR LOS RESIDUOS DIRECTAMENTE EN LOS BLOQUES DE UP Y DOWN
    """
    Class for the application of two 2D convolutions, each followed
    by a batch normalization layer and a ReLU layer (in that order)
    with residual connections.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    Returns:
        ResidualBlock (object): Instance of the ResidualBlock class.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv_2d = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding="same"),
            nn.BatchNorm2d(out_ch)
        )
        self.identity_map = nn.Sequential()
        if in_ch != out_ch:
            self.identity_map = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, padding="same"),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        add = self.double_conv_2d(x) + self.identity_map(x)
        output = self.relu(add)
        return output
    

class ResDownBlock(nn.Module):
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
        self.double_conv_2d = ResidualBlock(in_ch, out_ch)
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.double_conv_2d(x)
        x = self.pooling(skip)     
        return (x, skip)


class ResUpBlock(nn.Module):
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
        self.conv_transpose = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.double_conv_2d = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.conv_transpose(x)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv_2d(x)
        return x
    

class ResUNet(nn.Module):
    """
    UNet with residual connections between convolutions of the same block. Improvement
    over deep UNet implementations as reduce the problems of vanishing and exploding
    gradients.

    Args:
        in_ch (int, optional): Number of input channels (default: 1).
        out_classes (int, optional): Number of output classes. Use 1 for binary segmentation 
            (default: 1).
    """
    def __init__(self, in_ch = 1, out_classes = 1):
        super().__init__()
        # Encoder
        self.down1 = ResDownBlock(in_ch, 64)
        self.down2 = ResDownBlock(64, 128)
        self.down3 = ResDownBlock(128, 256)
        self.down4 = ResDownBlock(256, 512)
        # Bottleneck
        self.conv = ResidualBlock(512, 1024)
        # Decoder
        self.up1 = ResUpBlock(1024, 512)
        self.up2 = ResUpBlock(512, 256)
        self.up3 = ResUpBlock(256, 128)
        self.up4 = ResUpBlock(128, 64)
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
    


# Probar con UNet3Plus, AttentionResUNet y ResUNet3Plus
# AGResUNet (Attention Residual UNet) that incorporates both residual
# connections and attention gates to the basic UNet network. The implementation
# follows the paper Attention Gate ResU-Net for Automatic MRI Brain Tumor Segmentation
# (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9046011) which use
# additive attention.
class AttentionGate(nn.Module):
    """
    Class for the application of the attention gate.

    Args:
        k_ch: Number of key channels.
        g_ch: Number of gate channels.

    Returns:
        AttentionGate (object): Instance of the AttentionGate class.
    """
    def __init__(self, g_ch, k_ch):
        super().__init__()
        self.conv_gate = nn.Conv2d(g_ch, g_ch, 1)
        self.conv_key = nn.Sequential(
            nn.Conv2d(k_ch, k_ch, 1),
            nn.Conv2d(k_ch, g_ch, 2, 2),
        )
        self.attention = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(g_ch, 1, 1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2)
        )
        
    
    def forward(self, x, g):
        key = self.conv_key(x)
        gate = self.conv_gate(g)
        psi = self.attention(key + gate)
        output = psi * x

        return output

    

class AGResUNet(nn.Module):
    """
    UNet with residual connections and attention gates. Improvement
    over deep UNet implementations as reduce the problems of vanishing and exploding
    gradients with the residual connections and introduce attention mechanisms
    allowing the network to focus on relevant regions of the images.

    Args:
        in_ch (int, optional): Number of input channels (default: 1).
        out_classes (int, optional): Number of output classes. Use 1 for binary segmentation 
            (default: 1).
    """
    def __init__(self, in_ch = 1, out_classes = 1):
        super().__init__()
        # Encoder
        self.down1 = ResDownBlock(in_ch, 64)
        self.down2 = ResDownBlock(64, 128)
        self.down3 = ResDownBlock(128, 256)
        self.down4 = ResDownBlock(256, 512)
        # Bottleneck
        self.conv = ResidualBlock(512, 1024)
        # Decoder
        self.up1 = ResUpBlock(1024, 512)
        self.up2 = ResUpBlock(512, 256)
        self.up3 = ResUpBlock(256, 128)
        self.up4 = ResUpBlock(128, 64)
        # Attention gates
        self.att1 = AttentionGate(1024, 512)
        self.att2 = AttentionGate(512, 256)
        self.att3 = AttentionGate(256, 128)
        self.att4 = AttentionGate(128, 64)
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
        skip4 = self.att1(skip4, x)
        x = self.up1(x, skip4)
        skip3 = self.att2(skip3, x)
        x = self.up2(x, skip3)
        skip2 = self.att3(skip2, x)
        x = self.up3(x, skip2)
        skip1 = self.att4(skip1, x)
        x = self.up4(x, skip1)
        output = self.last_conv(x)

        return output