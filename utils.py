
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import os

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
    model.load_state_dict(checkpoint["state_dict"])

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
            
            