import os
import sys
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from BasicSR import SRDataset, Swin2SR, filter_120x120_images, visualize_super_resolve_dataloader, super_resolve_images

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up dataset and dataloader
hr_size = (120, 120)
lr_size = (60, 60)
dataset_path = r"..\data\SampleDataset"
image_paths = filter_120x120_images(dataset_path)
dataset = SRDataset(image_paths, lr_size, hr_size)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# initialise model
swin2sr_model = Swin2SR(img_size=60, patch_size=6, in_channels=3, embed_dim=768, depth=12, num_heads=12, scale_factor=2).to(device)

# load model weights if checkpoint is provided
def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# main function
def main(load=False, visualize=False, super_resolve=False, checkpoint_path=None):
    model = swin2sr_model

    # load model if specified
    if load and checkpoint_path:
        model = load_model(model, checkpoint_path)
        print("model loaded from checkpoint.")

    # visualise using existing function with psnr and ssim calculation
    if visualize:
        psnr, ssim, mse, lr, sr, hr = visualize_super_resolve_dataloader(dataloader, model, torch.nn.MSELoss(), device=device)
        print(f"Average PSNR: {psnr:.2f}, Average SSIM: {ssim:.2f}, Average MSE: {mse:.2f}")
        
        plt.figure(figsize=(10, 10))
        plt.subplot(131)
        plt.imshow(lr)
        plt.title("Low-res")
        plt.subplot(132)
        plt.imshow(sr)
        plt.title("Super-res")
        plt.subplot(133)
        plt.imshow(hr)
        plt.title("High-res")
        plt.show()


    # super-resolve images and save if specified
    if super_resolve:
        output_dir = "SR_Basic"
        super_resolve_images(dataloader, output_dir=output_dir, model=model, device=device)
        print(f"super-resolved images saved to {output_dir}")

# run main function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="satellite super-resolution with swin2sr")
    parser.add_argument("--load", action="store_true", help="load a trained model")
    parser.add_argument("--visualize", action="store_true", help="visualise low-res and high-res samples with metrics")
    parser.add_argument("--super_resolve", action="store_true", help="perform super-resolution on images")
    parser.add_argument("--checkpoint", type=str, help="path to model checkpoint if loading")
    
    args = parser.parse_args()
    main(load=args.load, visualize=args.visualize, super_resolve=args.super_resolve, checkpoint_path=args.checkpoint)
