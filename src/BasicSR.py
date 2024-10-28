import os
import torch
import rasterio
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import matplotlib.pyplot as plt
from transformers import Swin2SRImageProcessor, Swin2SRForImageSuperResolution
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=60, patch_size=6, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, num_patches_per_row, num_patches_per_col]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Swin2SR(nn.Module):
    def __init__(self, img_size=60, patch_size=6, in_channels=3, embed_dim=768, depth=12, num_heads=12, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        self.blocks = nn.ModuleList([SwinTransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.upscaled_patch_size = patch_size * scale_factor
        self.upsample = nn.Sequential(
            nn.Linear(embed_dim, (patch_size * scale_factor) ** 2 * in_channels),
            nn.GELU()
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.upsample(x)
        
        num_patches = (self.img_size // self.patch_size) ** 2
        x = x.view(B, num_patches, self.in_channels, self.upscaled_patch_size, self.upscaled_patch_size)

        patches_per_row = self.img_size // self.patch_size
        rows = []
        for i in range(patches_per_row):
            row_patches = []
            for j in range(patches_per_row):
                patch_idx = i * patches_per_row + j
                row_patches.append(x[:, patch_idx, :, :, :])
            row = torch.cat(row_patches, dim=3)
            rows.append(row)
        x = torch.cat(rows, dim=2)
        x = self.refine(x)

        return x

class SRDataset(Dataset):
    def __init__(self, data_paths, lr_size=(64, 64), hr_size=(128, 128)):
        self.data_paths = data_paths
        self.lr_size = lr_size
        self.hr_size = hr_size

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path = self.data_paths[idx]
        
        img = Image.open(image_path).convert('RGB')  
        
        img = np.array(img) 
        
        lr_img = resize(img, self.lr_size, anti_aliasing=True, preserve_range=True)  
        hr_img = resize(img, self.hr_size, anti_aliasing=True, preserve_range=True) 

        lr_img = lr_img / 255.0
        hr_img = hr_img / 255.0


        lr_tensor = torch.tensor(lr_img, dtype=torch.float32).permute(2, 0, 1)  # [3, 64, 64]
        hr_tensor = torch.tensor(hr_img, dtype=torch.float32).permute(2, 0, 1)  # [3, 128, 128]

        return lr_tensor, hr_tensor, image_path

def filter_120x120_images(datasetPath):
    imagePaths = []
    for root, dirs, files in os.walk(datasetPath):
        for file in files:
            image_path = os.path.join(root, file)
            with rasterio.open(image_path) as src:
                if src.width == 120 and src.height == 120:
                    imagePaths.append(image_path)
                else:
                    print(f"Skipping image {file} with size {src.width}x{src.height}")

    return imagePaths

def calculate_psnr(sr_img, hr_img):
    # PSNR calculation (using skimage for precision)
    sr_img_np = sr_img.cpu().detach().numpy()
    hr_img_np = hr_img.cpu().detach().numpy()
    psnr_value = peak_signal_noise_ratio(hr_img_np, sr_img_np, data_range=1.0)
    return psnr_value

def calculate_ssim(sr_img, hr_img):
    sr_img_np = sr_img.permute(1, 2, 0).cpu().detach().numpy()
    hr_img_np = hr_img.permute(1, 2, 0).cpu().detach().numpy()
    ssim_value = structural_similarity(hr_img_np, sr_img_np, multichannel=True, data_range=1.0, win_size=3) # need to explicitly state 3 because of the small image size
    return ssim_value

def visualize_super_resolve_dataloader(dataloader, vit_model, mse_criterion, device):
    psnr, ssim, mse = 0.0, 0.0, 0.0
    sample_count = 0

    for batch_idx, (lr_tensor, hr_tensor, _) in enumerate(tqdm(dataloader)):
        # move tensors to the specified device
        lr_tensor, hr_tensor = lr_tensor.to(device), hr_tensor.to(device)

        with torch.no_grad():
            # perform super-resolution on low-resolution input
            sr_image = vit_model(lr_tensor)

        # process each image in the batch
        for i in range(lr_tensor.size(0)): 
            # keep lr_img, sr_img, and hr_img as tensors for metric functions
            lr_img = lr_tensor[i].permute(1, 2, 0)
            sr_img = sr_image[i].permute(1, 2, 0)
            hr_img = hr_tensor[i].permute(1, 2, 0)

            # calculate metrics (assuming calculate_psnr and calculate_ssim handle tensors)
            psnr_value = calculate_psnr(sr_img, hr_img)
            ssim_value = calculate_ssim(sr_img, hr_img)
            mse_value = mse_criterion(sr_img, hr_img).item()

            # accumulate metrics and increment sample count
            psnr += psnr_value
            ssim += ssim_value
            mse += mse_value
            sample_count += 1

    # calculate and print average metrics
    print(f"Average PSNR: {psnr / sample_count:.2f}")
    print(f"Average SSIM: {ssim / sample_count:.4f}")
    print(f"Average MSE: {mse / sample_count:.4f}")

    # return average metrics and last images for further inspection if needed
    return psnr / sample_count, ssim / sample_count, mse / sample_count, lr_img.cpu().numpy(), sr_img.cpu().numpy(), hr_img.cpu().numpy()


def super_resolve_images(dataloader, output_dir, vit_model, device):
    os.makedirs(output_dir, exist_ok=True)
    
    vit_model.eval()
    with torch.no_grad():
        for batch_idx, (lr_tensor, hr_tensor, paths) in enumerate(tqdm(dataloader)):
            if lr_tensor.shape[1] == 1:
                lr_tensor = lr_tensor.repeat(1, 3, 1, 1)

            lr_tensor = lr_tensor.to(device)
            sr_images = vit_model(lr_tensor)

            for i in range(lr_tensor.shape[0]):
                # Convert and process images
                lr_img = lr_tensor[i].permute(1, 2, 0).cpu().numpy()
                sr_img = sr_images[i].permute(1, 2, 0).cpu().detach().numpy()

                lr_img = np.clip(lr_img, 0, 1)
                sr_img = np.clip(sr_img, 0, 1)

                # Display images
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(lr_img)
                plt.title(f"Low-Resolution: {lr_img.shape[:2]}")
                plt.subplot(1, 2, 2)
                plt.imshow(sr_img)
                plt.title(f"Super-Resolved: {sr_img.shape[:2]}")

                # Get the current path
                current_path = paths[i] if isinstance(paths[i], str) else paths[i][0]
                
                # Extract base filename while preserving numbers
                base_name = os.path.splitext(os.path.basename(current_path))[0]
                # base_name will be like 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_65'
                
                # Save with original numbers intact
                lr_save_path = os.path.join(output_dir, f"{base_name}_LR.png")
                sr_save_path = os.path.join(output_dir, f"{base_name}_SR.png")

                # Save images
                Image.fromarray((lr_img * 255).astype(np.uint8)).save(lr_save_path)
                Image.fromarray((sr_img * 255).astype(np.uint8)).save(sr_save_path)

                print(f"Saved images:\n  LR: {lr_save_path}\n  SR: {sr_save_path}")

    return lr_img, sr_img