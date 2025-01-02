import matplotlib.pyplot as plt
import numpy as np


def img_mask_plot(index, dataset):
    """
    Plot image and mask data side by side from a data set.

    Args:
        index (int): index of the image and mask in the data set.
        dataset (nn.Module): Pytorch Dataset object.
    """
    img, mask = dataset[index]

    plt.subplot(1,2,1)
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.axis("off")
    plt.title("Image")

    plt.subplot(1,2,2)
    plt.imshow(np.transpose(mask, (1,2,0)))
    plt.axis("off")
    plt.title("Mask")
    plt.show()


def plot_original_mask_pred(index, dataset, model, save=False, name=None):
    """
    Plot image and original and predicted masks data side by side from a data set.

    Args:
        index (int): index of the image and mask in the data set.
        dataset (nn.Module): Pytorch Dataset object.
        model (nn.Module): Model to use for predictions.
    """
    image = dataset[index][0]
    mask = dataset[index][1]
    prediction = model(dataset[index][0].unsqueeze(0))
    binary_mask = (prediction > 0.5).float()

    plt.subplot(1,3,1)
    plt.imshow(np.transpose(image, (1,2,0)))
    plt.axis("off")
    plt.title("Image")
    plt.subplot(1,3,2)
    plt.imshow(np.transpose(mask, (1,2,0)), cmap="gray")
    plt.axis("off")
    plt.title("Mask")
    plt.subplot(1,3,3)
    plt.imshow(binary_mask.squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("Predicted mask")
    if save:
        if name != None:
            plt.savefig(f"{index}_prediction_{name}.png")
    plt.show()


def plot_loss(history, save=False, name=None):
    """
    Plot training and validation loss history.

    Arg:
        history (pd.DataFrame): Pandas dataframe containing training
            and validation epoch losses.
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(8,4))
    plt.plot(np.log10(train_loss), color='blue')
    plt.plot(np.log10(val_loss), color='red')
    plt.title("Model loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train", "test"], loc="upper right")
    if save:
        if name != None:
            plt.savefig(f"loss_history_{name}.png")
    plt.show()