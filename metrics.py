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
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, target, epsilon=1):
        inputs = inputs.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (inputs * target).sum()
        dice = (2.*intersection + epsilon) / (inputs.sum() + target.sum() + epsilon)

        return 1 - dice