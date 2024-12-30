"""
Train and validation functions for the training of neural networks in Pytorch.
Based on https://www.kaggle.com/code/saeedghamshadzai/person-segmentation-deeplabv3-pytorch#Training-(Fine-Tuning).
"""

import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
from metrics import metrics


def train_loop(train_data, model, loss_fn, optimizer, device):
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
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        global_loss += loss.item() # REVISAR EN FUNCIÓN DE LA FUNCIÓN DE ERROR UTILIZADA
        train_bar.set_postfix(loss=loss.item())

    return global_loss


def validation_loop(validation_data, model, loss_fn, device):
    """
    Calculates the loss for the validation data.

    Args:
        validation_data (DataLoader): Pytorch DataLoader containing the validation
            images and masks.
        model (nn.Module): Pytorch model to evaluate.
    Returns:
        val_loss (float): Loss value of validation_data.
    """
    val_loss = 0.0

    for images, masks in validation_data:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        val_loss += loss.item()

    val_loss = val_loss / len(validation_data)
    return val_loss


def metrics_loop(validation_data, model, loss_fn, device):
    """
    Calculates the metrics for the validation data.

    Args:
        validation_data (DataLoader): Pytorch DataLoader containing the validation
            images and masks.
        model (nn.Module): Pytorch model to evaluate.
    Returns:
        val_metrics (list(float)): List of metrics for validation data.
        val_loss (float): Loss of the validation data.
    """
    val_loss = 0.0
    val_metrics = [0, 0, 0, 0]

    for images, masks in validation_data:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        
        temp_metrics = metrics(masks, outputs)
        val_metrics = [val_metrics[i] + temp_metrics[i] / len(validation_data) for i in range(len(val_metrics))]

        loss = loss_fn(outputs, masks)
        val_loss += loss.item()

    val_loss = val_loss / len(validation_data)
    return val_loss, val_metrics


def save_checkpoint(model_opt_state, file_name="checkpoint.pth.tar"):
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
    checkpoint = torch.load(filename, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def get_best_checkpoint(history, checkpoint_number=5):
    """
    Get the best checkpoint (lowest training loss and lowest validation loss).

    Args:
        history (pd.DataFrame): DataFrame containing the training history.
        top_number (int): Number of top validation losses to consider.
        checkpoint_number (int): Number of iterations from checkpoint to
            checkpoint.
    
    Returns:
        best (int): Index (0-based indexing) of the best epoch number.
    """
    checkpoints = history.iloc[-1:len(history.index):checkpoint_number] # -1 so that checkpoint index is epoch_index - 1
    checkpoints = checkpoints.sort_values(by=['val_loss', 'train_loss'], ascending=[True, True])
    return checkpoints.index[0]


def fit(train_data, validation_data, model, loss_fn, optimizer, epochs, checkpoint_number = 5, early_stop = 5, device="cpu",
        load_best=False): # REVISAR SI INTRODUCIR LEARNING RATE ADAPTATIVO
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
        stopping the training if there is no improvement in validation loss.

    Returns:
        history (pd.DataFrame): DataFrame containing:
            - "train_loss" (list(float)): List of training losses per epoch.
            - "val_loss" (list(float)): List of validation loss per epoch.
    """
    print("Starting the training...")
    train_loss = []
    val_loss = []
    counter = 0 # For early stopping

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}",)
        epoch_loss = train_loop(train_data, model, loss_fn, optimizer, device)
        epoch_loss = epoch_loss / len(train_data)

        epoch_val_loss = validation_loop(validation_data, model, loss_fn, device)

        train_loss.append(epoch_loss)
        val_loss.append(epoch_val_loss)
        print(f"Epoch training loss: {epoch_loss}. Epoch validation loss: {epoch_val_loss}")

        # Checkpoint if improvement (checked each 5 epochs)
        if (epoch+1) % checkpoint_number == 0:
            if (all(i >= epoch_loss for i in train_loss) or # REVISAR SI PONER OR O AND
                all(i >= epoch_val_loss for i in val_loss)):
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                filename = f"Epoch-{epoch+1}_checkpoint.pth.tar"
                save_checkpoint(checkpoint, file_name=filename)

        # Early stop
        if (all(i >= epoch_val_loss for i in val_loss)): # PUEDE SER MÁS EFICIENTE ALMACENAR EL VALOR CON MENOR ERROR, ASÍ NO HACE FALTA COMPROBAR EN CADA ITERACIÓN
            counter = 0
        elif counter > early_stop:
            print(f"No improvement in {early_stop} epochs. Stopping the trainig.")
            break
        else:
            counter += 1

        # Varying learning rate
        if epoch_val_loss < 0.1 and epoch_val_loss > 0.05:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        elif epoch_val_loss < 0.05:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    history = pd.DataFrame({
        "train_loss": train_loss,
        "val_loss": val_loss
    })
    
    if load_best:
        best = get_best_checkpoint(history, checkpoint_number)+1
        load_checkpoint(filename=f"Epoch-{best}_checkpoint.pth.tar", model=model, optimizer=optimizer)

    best_val_loss, best_metrics = metrics_loop(validation_data, model, loss_fn, device)
    print(f"""----------------Validation metrics----------------
|  Accuracy: {round(best_metrics[2], 3)}          Precision: {round(best_metrics[0], 3)}     |
--------------------------------------------------
|  Recall: {round(best_metrics[1], 3)}            Dice: {round(1-best_val_loss, 3)}          |
--------------------------------------------------
|  IoU: {round(best_metrics[3], 3)}                                    |
--------------------------------------------------""")

    print("...Training done!")
    
    return history