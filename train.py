import os
import time
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import SimpleITK as sitk
import torch
import torch.nn as nn

from init import Options
from Model import SRNet, Fusion_Net
from Brain_data import brainDataset, axialDataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_msssim import ssim

def main():
    opt = Options().parse()

    # Check the number of GPUs available
    if opt.gpu_ids != '-1':
        num_gpus = len(opt.gpu_ids.split(','))
    else:
        num_gpus = 0
    print('Number of GPUs:', num_gpus)

    # Create checkpoints directory
    if not os.path.exists(opt.save_dir):
        os.makedirs(f'{opt.save_dir}/')
    
    time_str = time.strftime("%y%m%d_%H%M", time.localtime())
    save_model_path = f'{opt.save_dir}/{time_str}'
    os.makedirs(save_model_path)

    # TensorBoard summary writer
    writer = SummaryWriter()

    # Load low-resolution and high-resolution file lists
    lr_files = sorted([f for f in os.listdir(opt.lr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    hr_files = sorted([f for f in os.listdir(opt.hr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    # Initialize networks
    net = SRNet(in_channels=1, num_features=16, growth_rate=16, num_blocks=2, num_layers=1).cuda()
    fusion_net = Fusion_Net(num_channels=2, num_features=16, growth_rate=16, num_blocks=2, num_layers=1).cuda(2)

    # Data parallelization
    net = nn.DataParallel(net, device_ids=[0, 1])
    fusion_net = nn.DataParallel(fusion_net, device_ids=[2, 3])

    print(f"Net is on device: {next(net.parameters()).device}")
    print(f"Fusion net is on device: {next(fusion_net.parameters()).device}")

    # Load pretrained models if available
    if opt.preload1 is not None:
        net.load_state_dict(torch.load(opt.preload1))
    if opt.preload2 is not None:
        fusion_net.load_state_dict(torch.load(opt.preload2))

    # Define loss function and optimizers
    loss_function = nn.L1Loss().cuda()
    net_optimizer = torch.optim.Adam(net.parameters(), opt.lr)
    fusion_net_optimizer = torch.optim.Adam(fusion_net.parameters(), opt.lr)

    # Training loop
    for epoch in range(opt.epochs):
        print("--" * 30)
        print(f"Epoch {epoch + 1}/{opt.epochs}")
        
        net.train()
        fusion_net.train()
        
        for f_i in range(len(lr_files)):
            lr_path = os.path.join(opt.lr_dir, lr_files[f_i])
            hr_path = os.path.join(opt.hr_dir, hr_files[f_i])
            print("--" * 10, lr_path)
            
            # Load coronal and sagittal datasets
            coronal_dataset = brainDataset(lr_path, hr_path, plane='coronal')
            sagittal_dataset = brainDataset(lr_path, hr_path, plane='sagittal')
            coronal_loader = DataLoader(coronal_dataset, batch_size=opt.batch_size, shuffle=False)
            sagittal_loader = DataLoader(sagittal_dataset, batch_size=opt.batch_size, shuffle=False)
            
            # Process coronal slices
            for i, (lr_coronal, _) in enumerate(coronal_loader):
                lr_cor = lr_coronal.cuda(net.device_ids[0])
                outputs_cor = net(lr_cor)

                if i == 0:
                    output_cor_3d = outputs_cor.squeeze(1)
                else:
                    output_cor_3d = torch.cat((output_cor_3d, outputs_cor.squeeze(1)), dim=0)

            cor_3d_np = output_cor_3d.detach().cpu().numpy()
            cor_3d_np = cor_3d_np.transpose(1, 0, 2)

            # Process sagittal slices
            for i, (lr_sagittal, _) in enumerate(sagittal_loader):
                lr_sag = lr_sagittal.cuda(net.device_ids[0])
                outputs_sag = net(lr_sag)

                if i == 0:
                    output_sag_3d = outputs_sag.squeeze(1)
                else:
                    output_sag_3d = torch.cat((output_sag_3d, outputs_sag.squeeze(1)), dim=0)

            sag_3d_np = output_sag_3d.detach().cpu().numpy()

            # Load axial dataset
            axial_dataset = axialDataset(cor_3d_np, sag_3d_np, hr_path)
            axial_loader = DataLoader(axial_dataset, batch_size=opt.batch_size, shuffle=False)

            # Process axial slices and train fusion network
            for lr_axial, hr_axial in axial_loader:
                lr_ax = lr_axial.cuda(fusion_net.device_ids[0])
                hr_ax = hr_axial.cuda(fusion_net.device_ids[0])
            
                fusion_output = fusion_net(lr_ax)
                loss = loss_function(fusion_output, hr_ax)

                # Backpropagation and optimization
                net_optimizer.zero_grad()
                fusion_net_optimizer.zero_grad()
                loss.backward()
                net_optimizer.step()
                fusion_net_optimizer.step()

            print(f"Epoch [{epoch+1}/{opt.epochs}], File: {f_i}/{len(lr_files)}, Loss: {loss.item():.4f}")
            writer.add_scalar('loss', loss.detach().item() / len(coronal_loader), epoch + 1)

        # Save model checkpoints
        if (epoch + 1) % opt.save_model_epochs == 0 or (epoch + 1) == opt.epochs:
            torch.save(net.state_dict(), f'{save_model_path}/RDN_Flair_{time_str}_last_{epoch + 1}.pth')

if __name__ == "__main__":
    main()
