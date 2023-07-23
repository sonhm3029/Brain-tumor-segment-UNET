from argparse import ArgumentParser
import os
import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from model import Unet
from utils import load_checkpoint, load_data,\
    train_epoch, val_epoch, save_checkpoint


def train(opt):
    device = torch.device(opt["device"])
    lr = opt["lr"]
    save_step = opt["save_period"]
    epochs = opt["epochs"]
    batch_size = opt["batch_size"]
    
    model = Unet(in_channels=3, out_channel=3)
    if opt["weights"] and os.path.exists(opt["weights"]):
        load_checkpoint(opt["weights"], model)
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Load data
    train_dataset, val_dataset = load_data(
        root=opt["dataset"],
        img_size=opt["img_size"]
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    
    
    
    start = time.time()
    for epoch in range(epochs):
        print(f"Training epoch {epoch} / {epochs}")
        train_epoch(model, epoch, train_loader, device, optimizer, loss_fn)
        print(f"Val epoch {epoch}/{epochs}")
        val_acc, dice_score = val_epoch(model, epoch, val_loader, device)
        if epoch % save_step ==0:
            save_checkpoint(model.state_dict(), epoch, val_acc, dice_score)
    end = time.time()
    print(f"Finished training after {end - start}")
    

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_period', type=int, default=50, help='Log model after every "save_period" epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Training learning reate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--dataset', type=str, default="dataset", help='Dataset root folder')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    
    opt = parser.parse_args()
    
    train(opt)