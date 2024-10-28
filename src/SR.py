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
import wandb

hr_size = 128 

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

        return lr_tensor, hr_tensor


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

datasetPath = r"SampleDataset"
imagePaths = filter_120x120_images(datasetPath)

dataset = SRDataset(imagePaths, lr_size=(64, 64), hr_size=(128, 128)) 
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)

num_epochs = 20

wandb.init(project="super_resolution", config={
    "learning_rate": 1e-4,
    "epochs": num_epochs,
    "batch_size": dataloader.batch_size,
})

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (lr_tensor, hr_tensor) in enumerate(progress_bar):
        lr_tensor, hr_tensor = lr_tensor.to(device), hr_tensor.to(device)

        optimizer.zero_grad()
        inputs = {'pixel_values': lr_tensor}
        outputs = model(**inputs)
        sr_image = outputs.reconstruction

        loss = criterion(sr_image, hr_tensor)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Log loss and images to W&B
        if batch_idx == 0:
            wandb.log({"Low-Resolution Input": wandb.Image(lr_tensor[0].permute(1, 2, 0).cpu().numpy())})
            wandb.log({"Super-Resolved Output": wandb.Image(sr_image[0].permute(1, 2, 0).cpu().detach().numpy())})

    wandb.log({"Epoch Loss": epoch_loss / len(dataloader)})

wandb.finish()
