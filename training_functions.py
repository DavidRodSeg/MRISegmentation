"""
Train and validation functions for the training of neural networks in Pytorch.
Based on https://www.kaggle.com/code/saeedghamshadzai/person-segmentation-deeplabv3-pytorch#Training-(Fine-Tuning).
"""

import torch
from torch import nn
from tqdm import tqdm
from metrics import DiceLossTorch
import pandas as pd


def train_loop(train_data, model, loss_fn, optimizer):
    """
    Mini-batch training of the model.

    Args:
        train_data (DataLoader): Pytorch dataloader containing the images
            and masks that will be used for training.
        model (nn.Module): Model that will be trained.
        loss_fn (function or nn.Module): Loss function that will be minimized
            during training.
        optimizer (function): Algorithm used for the training of the batch.

    Returns:
        global_loss (float): Total loss for the batches in train_data.
    """
    model.train()
    global_loss = 0.0
    train_bar = tqdm(train_data)
    for images, masks in train_bar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        global_loss += loss.item() # REVISAR EN FUNCIÓN DE LA FUNCIÓN DE ERROR UTILIZADA
        train_bar.set_postfix(loss=loss.item())

    return global_loss


def validation_loop(validation_data, model):
    """
    Calculates the Dice for the validation data.

    Args:
        validation_data (DataLoader): Pytorch DataLoader containing the validation
            images and masks.
        model (nn.Module): Pytorch model to evaluate.
    Returns:
        val_dice (float): Dice value of validation_data.
    """
    val_dice = 0.0
    for image, mask in validation_data:
        val_dice += DiceLossTorch(image, mask)

    return val_dice


def save_checkpoint(model_opt_state, file_name="checkpoint.pth"):
    """
    Saves the model and optimizer states.

    Args:
        model_opt_state (dict): Dictionary containing the model and optimizer
            states.
        file_name (str): Name of the file that will save the model and
            optimizer states.
    """
    torch.save(model_opt_state, file_name)
    print("Checkpoint saved")


def load_checkpoint(filename, model, optimizer):
    """
    Load the model and optimizer states previously saved.

    Args:
        filename (str): Name of the file containing the checkpoint.
        model (nn.Module): Pytorch model.
        optimizer (torch.optim.Optimizer): Pytorch optimizer.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def get_best_checkpoint(history, top_number = 5):
    """
    Get the best checkpoint (lowest training loss and top validation Dice).

    Args:
        history (pd.DataFrame): DataFrame containing the training history.
        top_number (int): Number of top validation Dice to consider.
    
    Returns:
        best (int): The best epoch number.
    """
    checkpoints = history.sort_values(by='val_dice', ascending=False).iloc[:top_number]
    checkpoints.sort_values(by='train_loss', ascending=True, inplace=True)
    best = checkpoints.index[0] + 1
    return best


def fit(train_data, validation_data, model, loss_fn, optimizer, epochs, early_stop = 5): # REVISAR SI INTRODUCIR LEARNING RATE ADAPTATIVO
    """
    Fit the model using the chosen loss and optimizer functions. Returns the best model
    obtained during training.

    Args:
        train_data (DataLoader): Pytorch DataLoader containing the images and masks for
            training.
        validation_data (DataLoader): Pytorch DataLoader containing the images and masks for
            validation.
        model (nn.Module): Pytorch model to be trained.
        loss_fn (nn.Moduel): Pytorch loss function to be minimized during training.
        optimizer (torch.optim.Optimizer): Pytorch optimizer to use during training.
        epochs (int): Maximum number of epochs to consider.
        early_stop (int): Number of epochs from the last checkpoint to wait before early 
        stopping the training if there is no improvement in validation Dice.

    Returns:
        history (pd.DataFrame): DataFrame containing:
            - "train_loss" (list(float)): List of training losses per epoch.
            - "val_dice" (list(float)): List of validation Dice per epoch.
    """
    print("Starting the training...")
    train_loss = []
    val_dice = []

    counter = 0 # For early stopping

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}",)
        epoch_loss = train_loop(train_data, model, loss_fn, optimizer)
        epoch_loss = epoch_loss / len(train_data)

        epoch_val_dice = validation_loop(validation_data, model)

        # Checkpoint if improvement (checked each 5 epochs)
        if epoch % 5 == 0:
            if (all(i < epoch_val_dice for i in val_dice) or # REVISAR SI PONER OR O AND
                all(i > epoch_loss for i in train_loss)):
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                filename = f"Epoch-{epoch+1}_checkpoint.pth"
                save_checkpoint(checkpoint, file_name=filename)

        # Early stop
        if (all(i < epoch_val_dice for i in val_dice)):
            counter = 0
        elif counter > early_stop:
            print(f"No improvement in {early_stop} epochs. Stopping the trainig.")
            break
        else:
            counter += 1

        train_loss.append(epoch_loss)
        val_dice.append(epoch_val_dice)
        print(f"Epoch training loss: {epoch_loss}. Epoch validation dice: {epoch_val_dice}")

    history = pd.DataFrame({
        "train_loss": train_loss,
        "val_dice": val_dice
    })
    
    best = get_best_checkpoint(history)
    load_checkpoint(filename=f"Epoch-{best}_checkpoint.pth", model=model, optimizer=optimizer)

    print("...Training done!")
    
    return history