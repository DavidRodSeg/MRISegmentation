import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, deeplabv3
from torch import nn
from torchinfo import summary
from tqdm import tqdm
from metrics import DiceLossTorch
from unet_blocks import UNet

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

# Base paths
dir = os.path.join(os.getcwd(), "old_data")
training_dir = os.path.join(dir, "training_data")
validation_dir = os.path.join(dir, "validation_data")
# Train data paths
train_flair_path = os.path.join(training_dir, "flair")
train_t1_path = os.path.join(training_dir, "t1")
train_t1ce_path = os.path.join(training_dir, "t1ce")
train_t2_path = os.path.join(training_dir, "t2")
train_masks_path = os.path.join(training_dir, "masks")
# Validation data paths
test_flair_path = os.path.join(validation_dir, "flair")
test_t1_path = os.path.join(validation_dir, "t1")
test_t1ce_path = os.path.join(validation_dir, "t1ce")
test_t2_path = os.path.join(validation_dir, "t2")
test_masks_path = os.path.join(validation_dir, "masks")

img_width = img_height = 240


#################### DATA AUGMENTATION ####################
transform = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.2),
    # A.RandomResizedCrop(size=[img_width, img_height], p=0.2,)
])

#################### DATA SET CREATION   ####################
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, masks_dir, transform=None):
        self.img_dir = img_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        masks_path = os.path.join(self.masks_dir, self.masks[index])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if np.max(image) > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        mask = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)
        if np.max(mask) > 1.0:
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            data = self.transform(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return image, mask

trainval_df = SegmentationDataset(train_flair_path, train_masks_path, transform=transform)
test_df = SegmentationDataset(test_flair_path, test_masks_path)

#################### DATA SET VISUALIZATION   ####################
def img_mask_plot(index, dataset):
    img, mask = dataset[index]

    plt.subplot(1,2,1)
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.axis('off')
    plt.title("Image")

    plt.subplot(1,2,2)
    plt.imshow(np.transpose(mask, (1,2,0)))
    plt.axis('off')
    plt.title("Mask")
    plt.show()

# for i in np.random.randint(0, len(trainval_df), 5):
#     img_mask_plot(i, trainval_df)


#################### TRAIN-VALIDATION SPLIT ####################
train_size = int(0.8*len(trainval_df))
val_size = len(trainval_df) - train_size
train_df, val_df = torch.utils.data.random_split(trainval_df, [train_size, val_size])

# for i in np.random.randint(0, len(val_df), 5):
#     img_mask_plot(i, val_df)

#################### DATALOADER CREATION   ####################
batch_size = 32
trainloader = DataLoader(train_df, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_df, batch_size=batch_size, shuffle=True)


#################### MODEL   ####################
model = UNet()


#################### MODEL TRAINING   ####################
# Hyperparameters
epochs = 1
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = DiceLossTorch()

def fit(train_data, model, loss_fn, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        train_bar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for images, masks in train_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() # REVISAR EN FUNCIÓN DE LA FUNCIÓN DE ERROR UTILIZADA
            train_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_data)}")

    return epoch_loss

# fit(train_data=trainloader, model=model, loss_fn=loss, optimizer=optimizer, epochs=epochs)

# torch.save(model.state_dict(), "model2_weights.pth")

model = UNet()
model.load_state_dict(torch.load("model2_weights.pth", weights_only=True))
model.eval()

def plot_original_mask_pred(index, dataset): # DEVUELVE TRANSFORMACIONES ALEATORIAS DEBIDO A QUE EJECUTA LA TRANSFORMACIÓN CON EL __getitem__
    prediction = model(dataset[index][0].unsqueeze(0))
    binary_mask = (prediction > 0.5).float()
    plt.subplot(1,3,1)
    plt.imshow(dataset[index][0].squeeze(0))
    plt.subplot(1,3,2)
    plt.imshow(dataset[index][1].squeeze(0), cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(binary_mask.squeeze(), cmap='gray')
    plt.show()

for i in np.random.randint(0, len(val_df), 5):
    plot_original_mask_pred(index=i, dataset=test_df)