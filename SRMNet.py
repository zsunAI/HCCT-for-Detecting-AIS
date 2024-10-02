import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
# data process
class NiiDataset(Dataset):
    def __init__(self, nii_file_5mm, nii_file_1mm, transform=None):
        self.nii_img_5mm = nib.load(nii_file_5mm).get_fdata()
        self.nii_img_1mm = nib.load(nii_file_1mm).get_fdata()
        self.transform = transform
        
        # 插值5mm到1mm
        z_factor = self.nii_img_1mm.shape[2] / self.nii_img_5mm.shape[2]
        self.nii_img_5mm_interp = zoom(self.nii_img_5mm, (1, 1, z_factor), order=3)
        
    def __len__(self):
        return self.nii_img_1mm.shape[2]
        
    def __getitem__(self, idx):
        coronal_slice_5mm = self.nii_img_5mm_interp[:, :, idx]
        sagittal_slice_5mm = self.nii_img_5mm_interp[idx, :, :]
        coronal_slice_1mm = self.nii_img_1mm[:, :, idx]
        sagittal_slice_1mm = self.nii_img_1mm[idx, :, :]
        sample = {'coronal_5mm': coronal_slice_5mm, 'sagittal_5mm': sagittal_slice_5mm,
                  'coronal_1mm': coronal_slice_1mm, 'sagittal_1mm': sagittal_slice_1mm}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    def __call__(self, sample):
        coronal_5mm, sagittal_5mm = sample['coronal_5mm'], sample['sagittal_5mm']
        coronal_1mm, sagittal_1mm = sample['coronal_1mm'], sample['sagittal_1mm']
        # Expand dimensions to fit the expected input of the model
        coronal_5mm = np.expand_dims(coronal_5mm, axis=0)
        sagittal_5mm = np.expand_dims(sagittal_5mm, axis=0)
        coronal_1mm = np.expand_dims(coronal_1mm, axis=0)
        sagittal_1mm = np.expand_dims(sagittal_1mm, axis=0)
        return {'coronal_5mm': torch.from_numpy(coronal_5mm).float(),
                'sagittal_5mm': torch.from_numpy(sagittal_5mm).float(),
                'coronal_1mm': torch.from_numpy(coronal_1mm).float(),
                'sagittal_1mm': torch.from_numpy(sagittal_1mm).float()}


# Residual Dense Block
class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.local_feature_fusion = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, 1, padding=0)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        out = torch.cat(features, 1)
        return self.local_feature_fusion(out) + x

# Pyramid Attention Network
class PANet(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(PANet, self).__init__()
        self.down1 = nn.Conv2d(in_channels, growth_rate, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(growth_rate, growth_rate, 3, stride=2, padding=1)
        self.attention = nn.Conv2d(growth_rate, in_channels, 1)
        self.up1 = nn.ConvTranspose2d(growth_rate, growth_rate, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(growth_rate, in_channels, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        down1 = F.relu(self.down1(x))
        down2 = F.relu(self.down2(down1))
        attention = torch.sigmoid(self.attention(down2))
        up1 = F.relu(self.up1(down2))
        up2 = F.relu(self.up2(up1 + down1 * attention))
        return x + up2 * attention

# Super-Resolution Network
class SRNet(nn.Module):
    def __init__(self, in_channels=1, growth_rate=64, num_rdb_blocks=3, num_rdb_layers=6):
        super(SRNet, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        
        # Add multiple RDB blocks
        self.rdb_blocks = nn.ModuleList()
        for _ in range(num_rdb_blocks):
            self.rdb_blocks.append(RDB(growth_rate, growth_rate, num_rdb_layers))
        
        # Add PANet for better feature aggregation
        self.panet = PANet(growth_rate, growth_rate)
        
        # Final convolutions
        self.conv1x1_1 = nn.Conv2d(growth_rate, growth_rate, 1)
        self.conv1x1_2 = nn.Conv2d(growth_rate, growth_rate, 1)
        self.conv1x1_3 = nn.Conv2d(growth_rate, growth_rate, 1)
        self.conv1x1_out = nn.Conv2d(growth_rate, in_channels, 1)

    def forward(self, x):
        out = F.relu(self.initial_conv(x))
        for rdb in self.rdb_blocks:
            out = rdb(out)
        out = self.panet(out)
        out = F.relu(self.conv1x1_1(out))
        out = F.relu(self.conv1x1_2(out))
        out = F.relu(self.conv1x1_3(out))
        return self.conv1x1_out(out)

def train_srnet(model, dataloader, optimizer, criterion, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, sample in enumerate(dataloader):
            coronal, sagittal = sample['coronal'], sample['sagittal']
            coronal, sagittal = coronal.to(device), sagittal.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            coronal_outputs = model(coronal)
            sagittal_outputs = model(sagittal)
            
            # Compute loss
            loss_coronal = criterion(coronal_outputs, coronal)
            loss_sagittal = criterion(sagittal_outputs, sagittal)
            loss = loss_coronal + loss_sagittal
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
    return model

def super_resolve_and_fuse(nii_file, model, output_path):
    dataset = NiiDataset(nii_file, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    sr_coronal_slices = []
    sr_sagittal_slices = []
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            coronal, sagittal = sample['coronal'], sample['sagittal']
            coronal, sagittal = coronal.to(device), sagittal.to(device)
            
            sr_coronal = model(coronal)
            sr_sagittal = model(sagittal)
            
            sr_coronal_slices.append(sr_coronal.cpu().numpy())
            sr_sagittal_slices.append(sr_sagittal.cpu().numpy())
    
    sr_coronal_3d = np.concatenate(sr_coronal_slices, axis=2)
    sr_sagittal_3d = np.concatenate(sr_sagittal_slices, axis=1)
    
    # Fusion operation (simple average for demonstration)
    fused_3d = (sr_coronal_3d + sr_sagittal_3d) / 2
    
    # Save the final super-resolved image
    nifti_img = nib.Nifti1Image(np.squeeze(fused_3d), np.eye(4))
    nib.save(nifti_img, output_path)

def train_srnet(model, dataloader, optimizer, criterion, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, sample in enumerate(dataloader):
            coronal_5mm, sagittal_5mm = sample['coronal_5mm'], sample['sagittal_5mm']
            coronal_1mm, sagittal_1mm = sample['coronal_1mm'], sample['sagittal_1mm']
            coronal_5mm, sagittal_5mm = coronal_5mm.to(device), sagittal_5mm.to(device)
            coronal_1mm, sagittal_1mm = coronal_1mm.to(device), sagittal_1mm.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            coronal_outputs = model(coronal_5mm)
            sagittal_outputs = model(sagittal_5mm)
            
            # Compute loss
            loss_coronal = criterion(coronal_outputs, coronal_1mm)
            loss_sagittal = criterion(sagittal_outputs, sagittal_1mm)
            loss = loss_coronal + loss_sagittal
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
    return model

    
# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
srnet = SRNet().to(device)

# Define optimizer and loss function
optimizer = optim.Adam(srnet.parameters(), lr=1e-4, betas=(0.5, 0.999))
criterion = nn.L1Loss()

# Load dataset
train_dataset = NiiDataset('path_to_5mm_ct_image.nii', 'path_to_1mm_ct_image.nii', transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Train the model
trained_srnet = train_srnet(srnet, train_loader, optimizer, criterion, num_epochs=25)

# Super-resolve and fuse the images
super_resolve_and_fuse('path_to_5mm_ct_image.nii', trained_srnet, 'super_resolved_image.nii')

