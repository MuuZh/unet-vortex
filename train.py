import torch.nn as nn
import torch.nn.functional as F
import torch
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from tqdm import tqdm , trange
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import os
import datetime
import time
import argparse


class SavedVortexDataset(Dataset):
    def __init__(self, data, brainmesh):
        self.data = data
        self.brainmesh = torch.FloatTensor(brainmesh)[None, :, :]  # [1, 175, 251]
    
    def __len__(self):
        return len(self.data['velocity'])
    
    def __getitem__(self, idx):
        label = torch.FloatTensor(self.data['label'][idx])
        
        label_onehot = torch.zeros((2, label.shape[0], label.shape[1]))
        label_onehot[0] = (label == 0).float()  
        label_onehot[1] = (label != 0).float()
        
        return {
            'velocity': torch.FloatTensor(self.data['velocity'][idx]),
            'vorticity': torch.FloatTensor(self.data['vorticity'][idx]),
            'ivd': torch.FloatTensor(self.data['ivd'][idx]),
            'label': label_onehot,
            'mask': self.brainmesh,
            'sub_id': self.data['sub_id'][idx],
            'frame': self.data['frame'][idx]
        }

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # based on  the input size of 175x251, the downsampled size will be:
        # 175x251 -> 87x125 -> 43x62 -> 21x31 -> 10x15
        
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = DoubleConv(16, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.up1 = DoubleConv(384, 128)
        self.up2 = DoubleConv(192, 64)
        self.up3 = DoubleConv(96, 32)
        self.up4 = DoubleConv(48, 16)
        self.outc = nn.Conv2d(16, 2, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # record the output of each layer to concatenate during upsampling
        x1 = self.inc(x)  # 175x251
        x2 = self.maxpool(x1)  # 87x125
        x2 = self.down1(x2)
        x3 = self.maxpool(x2)  # 43x62
        x3 = self.down2(x3)
        x4 = self.maxpool(x3)  # 21x31
        x4 = self.down3(x4)
        x5 = self.maxpool(x4)  # 10x15
        x5 = self.down4(x5)
        
        # use size parameter to specify the output size
        x = F.interpolate(x5, size=x4.shape[2:])
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x)
        x = F.interpolate(x, size=x3.shape[2:])
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = F.interpolate(x, size=x2.shape[2:])
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        x = F.interpolate(x, size=x1.shape[2:])
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)
        x = self.outc(x)
        return x

class MVUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity_unet = UNet(in_channels=2)
        self.vorticity_unet = UNet(in_channels=1)
        self.ivd_unet = UNet(in_channels=1)
        self.ensemble_unet = UNet(in_channels=6)  # 2*3=6 channels
        
    def forward(self, velocity, vorticity, ivd):
        # 3 views of
        vel_out = self.velocity_unet(velocity)
        vort_out = self.vorticity_unet(vorticity)
        ivd_out = self.ivd_unet(ivd)
        
        combined = torch.cat([vel_out, vort_out, ivd_out], dim=1)
        
        # ensemble prediction
        final_out = self.ensemble_unet(combined)
        return final_out
    

def prepare_training_data(dataset):
    # separate data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  
        pin_memory=True # use pinned memory for faster data transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )


    return train_loader, val_loader

def load_muvprocessed_data(filename, brainmesh):  
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return SavedVortexDataset(data, brainmesh)

from scipy.ndimage import binary_fill_holes
def gen_mesh_from_boundaries(boundaries):
   
    nan_mask = np.isnan(boundaries)

    boundaries[nan_mask] = 0


    filled = binary_fill_holes(boundaries).astype(int)


    filled[nan_mask] = 0

    mask = np.where(filled == 1, 1, 0)

    return mask

def parse_args():
    parser = argparse.ArgumentParser(description="MVU-Net vortex detection training")
    
    parser.add_argument("--data-path", default="prepre/train_data.pkl", 
                        help="path to processed data")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int, 
                        help="number of total epochs to train")
    parser.add_argument("--lr", default=1e-3, type=float, 
                        help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float,
                        help="weight decay")
    parser.add_argument("--print-freq", default=10, type=int,
                        help="print frequency")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int,
                        help="start epoch")
    parser.add_argument("--save-best", default=True, type=bool,
                        help="only save best model")
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use mixed precision training")

    args = parser.parse_args()
    return args

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    

    results_file = f"results{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    
    boundaries = np.load(Path('boundaries7parcellation.npy'))
    brainmesh = gen_mesh_from_boundaries(boundaries)
    
    muvdataset = load_muvprocessed_data(args.data_path, brainmesh)
    train_loader, val_loader = prepare_training_data(muvdataset)
    print(f"Data size: {len(muvdataset)}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    
    # create model
    model = MVUNet().to(device)
    
    # create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # mixed precision training
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    
    # resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    
    # record the best validation loss
    best_val_loss = float('inf')
    
    # training loop
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # one single epoch
        model.train()
        train_loss = 0
        train_batches = len(train_loader)
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch in train_bar:
            optimizer.zero_grad()
            
            # trsnfer batch to device
            velocity = batch['velocity'].to(device)
            vorticity = batch['vorticity'].to(device)
            ivd = batch['ivd'].to(device)
            labels = batch['label'].to(device)
            mask = batch['mask'].to(device)
            
            # mixed precision training
            if args.amp:
                with torch.cuda.amp.autocast():
                    outputs = model(velocity, vorticity, ivd)
                    loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
                    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(velocity, vorticity, ivd)
                loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches
        
        # validate the model
        model.eval()
        val_loss = 0
        val_batches = len(val_loader)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for batch in val_bar:
                velocity = batch['velocity'].to(device)
                vorticity = batch['vorticity'].to(device)
                ivd = batch['ivd'].to(device)
                labels = batch['label'].to(device)
                mask = batch['mask'].to(device)
                
                outputs = model(velocity, vorticity, ivd)
                loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                
                val_loss += loss.item()
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_batches
        
        # save results to file
        with open(results_file, "a") as f:
            f.write(f"[epoch: {epoch+1}]\n"
                   f"train_loss: {avg_train_loss:.4f}\n"
                   f"val_loss: {avg_val_loss:.4f}\n\n")
        
        # save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            if args.amp:
                save_dict['scaler'] = scaler.state_dict()
            
            torch.save(save_dict, os.path.join("save_weights", "best_model2.pth"))
    
    total_time = time.time() - start_time
    print(f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    args = parse_args()
    
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")
    
    main(args)
