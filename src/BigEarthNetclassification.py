import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import rasterio
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms

class_mapping = {
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",
    121: "Industrial or commercial units",
    122: "Road and rail networks and associated land",
    123: "Port areas",
    124: "Airports",
    131: "Mineral extraction sites",
    132: "Dump sites",
    133: "Construction sites",
    141: "Green urban areas",
    142: "Sport and leisure facilities",
    211: "Non-irrigated arable land",
    212: "Permanently irrigated land",
    213: "Rice fields",
    221: "Vineyards",
    222: "Fruit trees and berry plantations",
    223: "Olive groves",
    231: "Pastures",
    241: "Annual crops associated with permanent crops",
    242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture with significant areas of natural vegetation",
    244: "Agro-forestry areas",
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grassland",
    322: "Moors and heathland",
    323: "Sclerophyllous vegetation",
    324: "Transitional woodland-shrub",
    331: "Beaches, dunes, sands",
    332: "Bare rocks",
    333: "Sparsely vegetated areas",
    334: "Burnt areas",
    335: "Glaciers and perpetual snow",
    411: "Inland marshes",
    412: "Peat bogs",
    421: "Salt marshes",
    422: "Salines",
    423: "Intertidal flats",
    511: "Water courses",
    512: "Water bodies",
    521: "Coastal lagoons",
    522: "Estuaries",
    523: "Sea and ocean"
}

num_classes = len(class_mapping)

def getMappings():
    return class_mapping, num_classes

class SatelliteDataset(Dataset):
    def __init__(self, image_dir, reference_map_dir, transform=None, num_classes=num_classes, class_mapping=None):
        # Define and validate directories
        self.image_dir = os.path.abspath(image_dir)
        self.reference_map_dir = os.path.abspath(reference_map_dir)
        self.transform = transform
        self.num_classes = num_classes
        self.class_mapping = class_mapping if class_mapping else {}

        # Collect image file names
        self.image_files = [
            f for f in os.listdir(self.image_dir) if f.endswith('.png')
        ]

    def __len__(self):
        # Return the number of image files
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image file and patch ID
        image_file = self.image_files[idx]
        patch_id = os.path.splitext(image_file)[0]

        # Load the image
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        # Construct and load the reference map
        reference_map_filename = f"{patch_id}_reference_map.tif"
        #reference_map_path = os.path.join(self.reference_map_dir, reference_map_filename)
        reference_map_path = f"..\data\SampleDatasetReferenceMaps\{reference_map_filename}"

        #print(reference_map_path)
        #print(os.path.exists(reference_map_path))

        if os.path.exists(reference_map_path):
            with rasterio.open(reference_map_path) as src:
                reference_map = src.read(1)  # Read the first band only
            unique_classes = np.unique(reference_map)

            # Initialize multi-label vector
            multi_label_vector = np.zeros(self.num_classes, dtype=np.float32)
            for class_code in unique_classes:
                if class_code in self.class_mapping:
                    class_index = self.class_mapping[class_code]
                    multi_label_vector[class_index] = 1
        else:
            # Log missing file warning and return empty label vector
            print(f"Warning: Reference map not found for {patch_id}, returning empty labels.")
            multi_label_vector = np.zeros(self.num_classes, dtype=np.float32)

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(multi_label_vector, dtype=torch.float32), patch_id

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Unnormalization function (reversing the normalization applied earlier)
def unnormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img.clone()  # Clone the tensor to avoid modifying the original
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m) 
    return img

# Function to display the image
def imshow(img):
    img = unnormalize(img)     
    npimg = img.numpy()      
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def mapSample(image_path, reference_map_path):
    rgb_image = plt.imread(image_path) 
    rgb_image = rgb_image / np.max(rgb_image) 

    with rasterio.open(reference_map_path) as src:
        reference_map = src.read(1) 

    unique_classes = np.unique(reference_map)
    print(f"Unique class codes in the reference map: {unique_classes}")

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ref_map_plot = ax.imshow(reference_map, cmap='tab20')

    for class_code in unique_classes:
        pos = np.column_stack(np.where(reference_map == class_code))[0] 
        label = class_mapping.get(class_code, "Unknown")
        ax.text(pos[1], pos[0], label, color="white", fontsize=8, ha="center", va="center", bbox=dict(facecolor='black', alpha=0.5))

    plt.title("Reference Map with Land Cover Labels")
    fig.colorbar(ref_map_plot, ax=ax)

    plt.show()

def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():  # Disable gradient tracking
        for images, labels, id in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs) > 0.5  # Threshold at 0.5 for multi-label classification

            # Append predictions and labels to lists
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return all_preds, all_labels

# Calculate evaluation metrics
def calculate_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')  # Use 'macro' for multi-label
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def visualize_predictions(images, preds, labels):
    # Unnormalize and display images with predicted and true labels
    for i in range(min(5, len(images))):  # Show up to 5 images
        img = images[i].cpu()
        img = unnormalize(img)  # Unnormalize the image

        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert to (H, W, C)

        # Show predicted and true labels
        pred_labels = np.where(preds[i] == 1)[0]  # Get predicted classes
        true_labels = np.where(labels[i] == 1)[0]  # Get true classes
        plt.title(f"Pred: {pred_labels}, True: {true_labels}")
        plt.show()

def visualize_predictions_with_map(images, preds, labels, ids, reference_map_base_path, class_mapping):
    # Unnormalize and display images with predicted and true labels
    for i in range(min(5, len(images))):  # Show up to 5 images
        img = images[i].cpu()
        img = unnormalize(img)  # Unnormalize the image

        npimg = img.numpy()
        plt.figure(figsize=(12, 6))  # Adjust figure size for side-by-side visualization

        # Show the RGB image
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert to (H, W, C)
        pred_labels = np.where(preds[i] == 1)[0]  # Get predicted classes
        true_labels = np.where(labels[i] == 1)[0]  # Get true classes
        plt.title(f"Pred: {pred_labels}, True: {true_labels}")

        patch_id = ids[i]
        parent_folder = "_".join(patch_id.split("_")[:-2])
        reference_map_path = os.path.join(reference_map_base_path, parent_folder, patch_id, f"{patch_id}_reference_map.tif")

        if os.path.exists(reference_map_path):
            with rasterio.open(reference_map_path) as src:
                reference_map = src.read(1)

            plt.subplot(1, 2, 2)
            ref_map_plot = plt.imshow(reference_map, cmap='tab20')
            plt.colorbar(ref_map_plot)
            
            unique_classes = np.unique(reference_map)

            for class_code in unique_classes:
                pos = np.column_stack(np.where(reference_map == class_code))[0] 
                label = class_mapping.get(class_code, "Unknown")
                plt.text(pos[1], pos[0], label, color="white", fontsize=8, ha="center", va="center", bbox=dict(facecolor='black', alpha=0.5))

            plt.title(f"Reference Map with Land Cover Labels - {patch_id}")
        else:
            print(f"Reference map not found for {patch_id}. Skipping reference map visualization.")
        
        plt.show()