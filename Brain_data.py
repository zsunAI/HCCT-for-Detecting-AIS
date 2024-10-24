import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomAffine
from PIL import Image

class SynchronizedTransform:
    """
    SynchronizedTransform ensures that two images (e.g., low-resolution and high-resolution slices) 
    undergo the same transformations using the same random seed.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2):
        seed = np.random.randint(2147483647)  # Use the same seed for both images
        torch.manual_seed(seed)
        img1 = self.transform(img1)
        torch.manual_seed(seed)
        img2 = self.transform(img2)
        return img1, img2

class BrainDataset(Dataset):
    """
    BrainDataset handles loading and preprocessing of brain CT data.
    It loads low-resolution and high-resolution images, extracts slices based on the given plane,
    and applies synchronized transformations.
    """
    def __init__(self, lr_dir, hr_dir, plane='coronal'):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.plane = plane

        # Load the low-resolution and high-resolution images
        self.lr_img = nib.load(self.lr_dir).get_fdata()
        self.hr_img = nib.load(self.hr_dir).get_fdata()

        # Get slices based on the specified plane
        self.lr_slices, self.hr_slices = self._get_slices()

        # Define the transformations
        self.transforms = SynchronizedTransform(Compose([
            ToTensor(),
            Normalize(mean=[0.0], std=[1.0]),  # Normalize to [0, 1]
            # Uncomment the following lines to add more data augmentation transformations
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            # RandomRotation(degrees=30),
            # RandomAffine(degrees=0, scale=(0.8, 1.2))
        ]))

    def _get_slices(self):
        """
        Extracts slices from the 3D data based on the specified plane.
        """
        if self.plane == 'coronal':
            lr_slices = [self.lr_img[:, i, :] for i in range(self.lr_img.shape[1])]
            hr_slices = [self.hr_img[:, i, :] for i in range(self.hr_img.shape[1])]
        elif self.plane == 'sagittal':
            lr_slices = [self.lr_img[i, :, :] for i in range(self.lr_img.shape[0])]
            hr_slices = [self.hr_img[i, :, :] for i in range(self.hr_img.shape[0])]
        elif self.plane == 'axial':
            lr_slices = [self.lr_img[:, :, i] for i in range(self.lr_img.shape[2])]
            hr_slices = [self.hr_img[:, :, i] for i in range(self.hr_img.shape[2])]
        else:
            raise ValueError("Plane must be 'coronal', 'sagittal', or 'axial'")

        return lr_slices, hr_slices

    def __len__(self):
        """
        Returns the total number of slices.
        """
        return len(self.lr_slices)

    def __getitem__(self, idx):
        """
        Returns a pair of low-resolution and high-resolution slices with applied transformations.
        """
        lr_slice = self.lr_slices[idx]
        hr_slice = self.hr_slices[idx]

        # Convert slices to float32
        lr_slice = lr_slice.astype(np.float32)
        hr_slice = hr_slice.astype(np.float32)

        # Apply synchronized transformations
        lr_slice, hr_slice = self.transforms(lr_slice, hr_slice)

        return lr_slice, hr_slice


class AxialDataset(Dataset):
    """
    AxialDataset handles loading and preprocessing of axial brain CT data.
    It combines coronal and sagittal slices, extracts corresponding high-resolution slices,
    and applies synchronized transformations.
    """
    def __init__(self, cor_np, sag_np, hr_nii_path):
        self.cor_img = cor_np
        self.sag_img = sag_np
        self.hr_img = nib.load(hr_nii_path).get_fdata()

        # Get slices for the combined coronal and sagittal images, and high-resolution images
        self.lr_slices, self.hr_slices = self._get_slices()

        # Define the transformations
        self.transforms = SynchronizedTransform(Compose([
            ToTensor(),
            Normalize(mean=[0.0], std=[1.0]),  # Normalize to [0, 1]
            # Uncomment the following lines to add more data augmentation transformations
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            # RandomRotation(degrees=30),
            # RandomAffine(degrees=0, scale=(0.8, 1.2))
        ]))

    def _get_slices(self):
        """
        Combines coronal and sagittal slices and extracts corresponding high-resolution slices.
        """
        result = []
        for j in range(self.cor_img.shape[2]):
            slice_A = self.cor_img[:, :, j]   
            slice_B = self.sag_img[:, :, j] 

            combined = np.stack((slice_A, slice_B), axis=0)  # Combine in the first dimension
            result.append(combined)

        lr_slices = np.array(result)  # Shape: [batchsize, 2, x, y]
        hr_slices = np.array([self.hr_img[:, :, i][np.newaxis, :] for i in range(self.hr_img.shape[2])])  # Shape: [batchsize, 1, x, y]
        
        return lr_slices, hr_slices

    def __len__(self):
        """
        Returns the total number of slices.
        """
        return len(self.lr_slices)

    def __getitem__(self, idx):
        """
        Returns a pair of low-resolution and high-resolution slices with applied transformations.
        """
        lr_slice = self.lr_slices[idx]
        hr_slice = self.hr_slices[idx]

        # Convert slices to float32
        lr_slice = lr_slice.astype(np.float32)
        hr_slice = hr_slice.astype(np.float32)

        # Apply synchronized transformations
        lr_slice, hr_slice = self.transforms(lr_slice, hr_slice)
        
        # Permute dimensions to match expected shape
        lr_slice = lr_slice.permute(1, 0, 2)  
        hr_slice = hr_slice.permute(1, 0, 2)  
        
        return lr_slice, hr_slice
