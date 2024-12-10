import torchvision
import os
import shutil


output_dir = "C:/Users/anees/Documents/6.5940_final/flowers102_imagefolder"
# Load the dataset
train_dataset = torchvision.datasets.Flowers102(root="C:/Users/anees/Documents/6.5940_final", split="train", download=True)
val_dataset = torchvision.datasets.Flowers102(root="C:/Users/anees/Documents/6.5940_final", split="val", download=False)

# Helper function to organize images into folders
def organize_split(dataset, split_name):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    images = dataset._image_files
    labels = dataset._labels  # Class labels (1-based index)
    
    for label in set(labels):
        label_dir = os.path.join(split_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

    for img_path, label in zip(images, labels):
        class_folder = os.path.join(split_dir, str(label))
        dst_path = os.path.join(class_folder, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)

# Organize training and validation splits
organize_split(train_dataset, "train")
organize_split(val_dataset, "val")

print(f"Dataset successfully converted to ImageFolder format at {output_dir}")