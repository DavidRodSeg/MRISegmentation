"""
Metrics and loss functions used during training and validation.
"""

import numpy as np
import torch


class DiceLossTorch(torch.nn.Module):
    """
    Calculate the Dice loss function: 1 - Dice.

    Args:
        inputs (array): Input image.
        target (array): Mask to compare with the image.

    Returns:
        loss (float): 1 - Dice.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, epsilon=1):
        inputs = inputs.flatten()
        target = target.flatten()

        inputs = inputs.float()
        target = target.float()

        intersection = torch.sum(inputs * target)
        dice = (2.0 * intersection + epsilon) / (torch.sum(inputs) + torch.sum(target) + epsilon)

        return 1 - dice
    
def precision(real, prediction):
    """
    Calculate the precision of a prediction.

    Args:
        real (torch.Tensor): Tensor representing the ground truth image.
        prediction (torch.Tensor): Tensor representing the predicted image.

    Returns:
        float: Precision value.
    """
    intersection = np.sum(real*prediction)
    return intersection / np.sum(prediction)
    

def recall(real, prediction):
    """
    Calculate the recall of a prediction.

    Args:
        real (torch.Tensor): Tensor representing the ground truth image.
        prediction (torch.Tensor): Tensor representing the predicted image.

    Returns:
        float: Recall value.
    """
    intersection = np.sum(real*prediction)
    return intersection / np.sum(real)


def jaccard_index(real, prediction):
    """
    Calculate the Jaccard index (IoU) of a prediction.

    Args:
        real (torch.Tensor): Tensor representing the ground truth image.
        prediction (torch.Tensor): Tensor representing the predicted image.

    Returns:
        float: IoU value.
    """
    intersection = np.sum(real*prediction)
    union = np.sum(real) + np.sum(prediction) - intersection
    return intersection / union

def accuracy(real, prediction):
    """
    Calculate the accuracy of a prediction.

    Args:
        real (torch.Tensor): Tensor representing the ground truth image.
        prediction (torch.Tensor): Tensor representing the predicted image.

    Returns:
        float: Accuracy value.
    """
    true_values = np.sum(real==prediction)
    false_values = np.sum(real!=prediction)
    return true_values / (true_values + false_values)

def metrics(real, prediction):
    """
    Calculate the following metrics: accuracy, precision, Jaccard index, and recall.

    Args:
        real (torch.Tensor): Tensor representing the ground truth image.
        prediction (torch.Tensor): Tensor representing the predicted image.

    Returns:
        list[float]: A list containing the calculated metrics in the following order:
                    [precision, recall, accuracy, IoU].
    """
    real = real.detach().numpy()
    prediction = np.round(prediction.detach().numpy())
    TP = np.sum((real == 1.) & (prediction == 1.)) # & is the bitwise logical operator used for element wise operations in NumPy
    FP = np.sum((real == 0.) & (prediction == 1.))
    TN = np.sum((real == 0.) & (prediction == 0.))
    FN = np.sum((real == 1.) & (prediction == 0.))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    IoU = TP / (TP + FP + FN)

    return [precision, recall, accuracy, IoU]