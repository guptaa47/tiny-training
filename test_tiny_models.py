import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from compilation.convert import (
    build_quantized_mcunet,
    build_quantized_mbv2,
    build_quantized_proxyless
)


def get_model(model_name="mcunet"):
    # some configs
    rs = 128
    num_classes = 10
    int8_bp = True

    # convert pytorch model to forward graph
    if model_name == "mbv2":
        model, _ = build_quantized_mbv2(num_classes=num_classes)
    elif model_name == "mcunet":
        model, _ = build_quantized_mcunet(num_classes=num_classes)
    elif model_name == "proxyless":
        model, _ = build_quantized_proxyless(num_classes=num_classes)
    
    return model

# Define the function to evaluate the model
def evaluate_model(model_path, test_data_dir, batch_size, num_workers):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    print("Loading model...")
    model.eval()  # Set the model to evaluation mode

    # Define the transformation for the test set
    transform = transforms.Compose([
        transforms.Resize(144),  # Resize the shorter side to 256 pixels
        transforms.CenterCrop(128),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalize to ImageNet means
            std=[0.229, 0.224, 0.225],  # Normalize to ImageNet stds
        ),
    ])

    # Load the ImageNet test dataset
    print("Loading test dataset...")
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Define the metrics
    top1_correct = 0
    top5_correct = 0
    total = 0

    # Evaluate the model
    print("Evaluating model...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get top-1 and top-5 predictions
            _, top1_pred = outputs.topk(1, dim=1)
            _, top5_pred = outputs.topk(5, dim=1)

            # Update metrics
            total += labels.size(0)
            top1_correct += (top1_pred.squeeze(1) == labels).sum().item()
            top5_correct += sum(labels[i] in top5_pred[i] for i in range(labels.size(0)))

            if i % 10 == 0:
                print(f"Processed {i * batch_size}/{len(test_dataset)} images...")

    # Calculate accuracies
    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total

    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    val_data_dir = os.path.join(os.getenv("IMAGENET_PATH"), 'val')

    evaluate_model(
        model=model
        test_data_dir=val_data_dir,
        batch_size=1,
        num_workers=4,
    )
