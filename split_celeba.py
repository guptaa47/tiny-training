import os
import shutil
import random

def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    """
    Splits the dataset into train and validation sets.

    Parameters:
    - source_dir (str): Path to the source directory containing subfolders for each class.
    - dest_dir (str): Path to the destination directory where train and validation sets will be saved.
    - split_ratio (float): Ratio of data used for training (0 to 1). The remaining will be used for validation.
    """

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Define paths for train and validation sets
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate over each class (subfolder in the source directory)
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        # Skip if not a directory (i.e., it's not a class folder)
        if not os.path.isdir(class_path):
            continue

        # Create subdirectories for the train and validation sets in the destination
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get the list of image files in the current class folder
        image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(image_files)

        # Split the images into train and validation sets
        split_index = int(len(image_files) * split_ratio)
        train_files = image_files[:split_index][:100]
        val_files = image_files[split_index:][:50]

        # Move the files to the corresponding directories
        for file in train_files:
            shutil.move(os.path.join(class_path, file), os.path.join(train_class_dir, file))

        for file in val_files:
            shutil.move(os.path.join(class_path, file), os.path.join(val_class_dir, file))

    print(f"Dataset split into 'train' and 'val' sets with a {split_ratio * 100}% training split.")

source_directory = '/home/gridsan/agupta2/6.5940/tiny_imagenet'
destination_directory = '/home/gridsan/agupta2/6.5940/mini_imagenet_split'
split_dataset(source_directory, destination_directory, split_ratio=0.8)
