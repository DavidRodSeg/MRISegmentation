import numpy as np
# import tensorflow as tf
import torch

# def dice_coefficient(y_true, y_pred, smooth=1):
#     y_true_f = tf.keras.flatten(y_true)
#     y_pred_f = tf.keras.flatten(y_pred)
#     intersection = tf.keras.sum(y_true_f * y_pred_f)
#     union = tf.keras.sum(y_true_f) + tf.keras.sum(y_pred_f)
#     return (2. * intersection + smooth) / (union + smooth)

# def dice_coefficient_numpy(y_true, y_pred, smooth=1):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     union = np.sum(y_true_f) + np.sum(y_pred_f)
#     return (2. * intersection + smooth) / (union + smooth)

# def jaccard_index(y_true, y_pred, smooth=100):
#     """Calculates the Jaccard index (IoU), useful for evaluating the model's performance."""
#     y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])  # Flatten and cast ground truth
#     y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])  # Flatten and cast predictions
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)  # Compute intersection
#     total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection  # Total pixels
#     return (intersection + smooth) / (total + smooth)

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
    """
    intersection = np.sum(real*prediction)
    return intersection / np.sum(prediction)
    

def recall(real, prediction):
    """
    Calculate the recall of a prediction.
    """
    intersection = np.sum(real*prediction)
    return intersection / np.sum(real)


def jaccard_index(real, prediction):
    """
    Calculate the Jaccard index (IoU) of a prediction.
    """
    intersection = np.sum(real*prediction)
    union = np.sum(real) + np.sum(prediction) - intersection
    return intersection / union

def accuracy(real, prediction):
    """
    Calculate the accuracy of a prediction.
    """
    true_values = np.sum(real==prediction)
    false_values = np.sum(real!=prediction)
    return true_values / (true_values + false_values)

def metrics(real, prediction):
    """
    Calculate the following metrics: accuracy, precision, jaccard index,
    recall.
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