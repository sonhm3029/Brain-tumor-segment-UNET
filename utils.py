
from tqdm import tqdm
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import BrainTumorDataset

writer = SummaryWriter("runs")

def save_checkpoint(state, epoch, val_acc, dice_scrore, save_folder="runs/checkpoint"):
    
    print(f"""=> Saving checkpoint at epoch {epoch}:
                \n Validation accuracy: {val_acc}
                \n Validation dice score: {dice_scrore}
          """)
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    torch.save(state, f"{save_folder}/model_epoch_{epoch}.pth.tar")

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(torch.load(checkpoint))
    
    

def load_data(root="dataset", img_size=224):
    
    train_transforms = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    train_dataset = BrainTumorDataset(root=root, part="train", transforms=train_transforms)
    val_dataset = BrainTumorDataset(root=root, part="valid", transforms=val_transforms)
    
    return train_dataset, val_dataset

def train_epoch(net, epoch, dataloader,
                device, optimizer, criterion):
    net.train() # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        images, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 100 mini-batches
            writer.add_scalar('Training loss',
                                running_loss / 10,
                                epoch * len(dataloader) + i
                                )
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

def val_epoch(net, epoch, dataloader, device):
    net.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            
            preds = torch.sigmoid(net(images))
            preds = (preds > 0.5).float()
            
            num_correct += (preds == labels).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2* (preds * labels).sum())/((preds + labels).sum() + 1e-8)
            
    val_acc = num_correct*100/num_pixels
    dice_score = dice_score / len(dataloader)
    print(f"Valdation result: {num_correct}/{num_pixels} with acc {val_acc:.2f}")
    print(f"Dice score: {dice_score/len(dataloader)}")
    writer.add_scalar("Validation Accuracy",
                      val_acc,
                      epoch)
    writer.add_scalar("Dice score",
                      dice_score,
                      epoch)
            
    return val_acc, dice_score
            