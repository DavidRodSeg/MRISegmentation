import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, deeplabv3
from torch import nn
from tqdm import tqdm
from metrics import DiceLossTorch
from training_functions import fit
from plot_functions import img_mask_plot, plot_original_mask_pred, plot_loss



#################### GLOBAL CONSTANTS AND PATHS ####################
def set_seed(seed):
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): Seed for random methods.
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
in_ch = 1


#################### DATA AUGMENTATION ####################
transform = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.2),
    # A.RandomResizedCrop(size=(img_width, img_height), p=0.2,),
    A.GaussianBlur(p=1.0)
])

def min_max_normalization(image):
    min_value = image.min()
    max_value = image.max()

    return (image - min_value) / (max_value - min_value)


#################### DATA SET CREATION   ####################
class SegmentationDataset(Dataset):
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

trainval_df = SegmentationDataset(train_flair_path, train_masks_path, transform=transform, in_ch=in_ch)
test_df = SegmentationDataset(test_flair_path, test_masks_path, in_ch=in_ch)


#################### DATA SET VISUALIZATION   ####################
for i in np.random.randint(0, len(trainval_df), 5):
    img_mask_plot(i, trainval_df)


#################### TRAIN-VALIDATION SPLIT ####################
train_size = int(0.8*len(trainval_df))
val_size = len(trainval_df) - train_size
train_df, val_df = torch.utils.data.random_split(trainval_df, [train_size, val_size])


#################### DATALOADER CREATION   ####################
batch_size = 32
trainloader = DataLoader(train_df, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_df, batch_size=batch_size, shuffle=True)


#################### PRETRAINED MODEL   ####################
class Pretrained_Model(nn.Module):
    """
    Defines the fine-tuned model of the DeepLabV3 pretrained model.
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.convolution = nn.Conv2d(in_ch, 3, (3,3), padding="same")
        self.pretrained_model = deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.DEFAULT)
        self.pretrained_model.classifier = deeplabv3.DeepLabHead(2048, 1)
        self.pretrained_model.aux_classifier = None
        self.sigmoid = nn.Sigmoid()

        for name, param in self.pretrained_model.backbone.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False  

    def forward(self, x):
        x = self.convolution(x)
        x = self.pretrained_model(x)["out"] # Shape: (batch_size, num_classes, height, width)
        output = self.sigmoid(x)

        return output
    
model = Pretrained_Model(in_ch=in_ch)
device = "cpu" #'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


#################### MODEL TRAINING   ####################
# Hyperparameters
epochs = 5
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = DiceLossTorch()

history = fit(train_data=trainloader, validation_data=valloader, model=model, loss_fn=loss, optimizer=optimizer,
               epochs=epochs, device=device)
plot_loss(history)


#################### MODEL TESTING   ####################
model.eval()
for i in np.random.randint(0, len(val_df), 5):
    plot_original_mask_pred(index=i, dataset=test_df, model=model)