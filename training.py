import os
import random
import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader
from utils import DiceLossTorch
from models import UNet, ResUNet, AGResUNet, Pretrained_Model
from utils import fit, img_mask_plot, plot_original_mask_pred, plot_loss
from data_set_class.segmentation_data_set import SegmentationDataset



#################### GLOBAL CONSTANTS AND PATHS ####################
def set_seed(seed):
    """
    Set all random seeds for reproducibility.
    """
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed = 300
set_seed(seed)
g = torch.Generator()
g.manual_seed(seed) # DataLoader seed for reproducibility

# Train data paths
train_path = os.path.join(os.getcwd(), "training/images")
train_masks_path = os.path.join(os.getcwd(), "training/masks")

# Validation data paths
test_path = os.path.join(os.getcwd(), "validation/images")
test_masks_path = os.path.join(os.getcwd(), "validation/masks")

img_width = img_height = 240
in_ch = 4


#################### DATA AUGMENTATION ####################
transform = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.2),
    # A.RandomResizedCrop(size=(img_width, img_height), p=0.2,),
    A.GaussianBlur(p=1.0)
])


#################### DATA SET CREATION ####################
trainval_df = SegmentationDataset(train_path, train_masks_path, transform=transform, in_ch=in_ch)
test_df = SegmentationDataset(test_path, test_masks_path, in_ch=in_ch)


#################### DATA SET VISUALIZATION ####################
for i in np.random.randint(0, len(trainval_df), 5):
    img_mask_plot(i, trainval_df)


#################### TRAIN-VALIDATION SPLIT ####################
train_size = int(0.8*len(trainval_df))
val_size = len(trainval_df) - train_size
train_df, val_df = torch.utils.data.random_split(trainval_df, [train_size, val_size])


#################### DATALOADER CREATION ####################
batch_size = 8
trainloader = DataLoader(train_df, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_df, batch_size=batch_size, shuffle=True)


#################### MODEL ####################
unet_model = UNet(in_ch=in_ch)
deeplab_model = Pretrained_Model(in_ch=in_ch)
device = "cuda" if torch.cuda.is_available() else "cpu"
unet_model.to(device)
deeplab_model.to(device)


#################### MODEL TRAINING ####################
# Hyperparameters
epochs = 20
learning_rate = 1e-4
optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss = DiceLossTorch()

unet_history = fit(train_data=trainloader, validation_data=valloader, model=unet_model, loss_fn=loss, optimizer=optimizer,
               epochs=epochs, device=device)
plot_loss(unet_history)

batch_size = 16
deeplab_history = fit(train_data=trainloader, validation_data=valloader, model=unet_model, loss_fn=loss, optimizer=optimizer,
               epochs=epochs, device=device)
plot_loss(deeplab_history)


#################### MODEL TESTING ####################
unet_model.eval()
for i in np.random.randint(0, len(val_df), 10):
    plot_original_mask_pred(index=i, dataset=test_df, model=unet_model)

deeplab_model.eval()
for i in np.random.randint(0, len(val_df), 10):
    plot_original_mask_pred(index=i, dataset=test_df, model=deeplab_model)