"""
A PyTorch Dataset class designed for image segmentation. It includes a Min-Max normalization function,
which is utilized within the class.
"""

import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


def min_max_normalization(image):
    min_value = image.min()
    max_value = image.max()

    return (image - min_value) / (max_value - min_value)


class SegmentationDataset(Dataset):
    """
    A PyTorch Dataset class designed for image segmentation.

    Args:
        img_dir (str): Path to the directory containing input images.
        masks_dir (str): Path to the directory containing corresponding segmentation masks.
        transform (callable, optional): Optional transformations to be applied to the images and masks.
        in_ch (int, optional): Number of input channels. Default is 1.
    """
    def __init__(self, img_dir, masks_dir, transform=None, in_ch = 1):
        self.img_dir = img_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.transform = transform
        self.in_ch = in_ch

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        masks_path = os.path.join(self.masks_dir, self.masks[index])

        if self.in_ch == 1:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image, axis=2)
        elif self.in_ch == 3:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        elif self.in_ch == 4:
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if np.max(image) > 1.0:
            image = image.astype(np.float32)
            image = min_max_normalization(image)

        mask = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)
        if np.max(mask) > 1.0:
            mask = mask.astype(np.float32)
            mask = min_max_normalization(mask)
        else:
            mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=2)

        if self.transform:
            data = self.transform(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return image, mask