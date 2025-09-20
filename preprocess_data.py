"""
Data preprocessing script for the TU Berlin dataset.
This script loads the raw dataset, applies transformations, and saves the preprocessed data.
"""

from pathlib import Path
import pickle
import torch
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

def preprocess_and_save_dataset():
    """Preprocess the TU Berlin dataset and save to disk."""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Loading TU Berlin dataset...")
    ds = load_dataset("kmewhort/tu-berlin-png")
    
    # Data transforms (same as in main.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Process training data
    print("Processing training data...")
    train_images = []
    train_labels = []
    
    for example in tqdm(ds["train"], desc="Processing train set"):
        # Convert to RGB and apply transform
        image = example["image"].convert("RGB")
        transformed_image = transform(image)
        
        train_images.append(transformed_image)
        train_labels.append(example["label"])
    
    # Process test data
    print("Processing test data...")
    test_images = []
    test_labels = []
    
    for example in tqdm(ds["test"], desc="Processing test set"):
        # Convert to RGB and apply transform
        image = example["image"].convert("RGB")
        transformed_image = transform(image)
        
        test_images.append(transformed_image)
        test_labels.append(example["label"])
    
    # Convert to tensors
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    
    # Save training data
    train_data = {
        "images": train_images,
        "labels": train_labels,
        "num_classes": ds["train"].features["label"].num_classes
    }
    
    with open(data_dir / "train_preprocessed.pkl", "wb") as f:
        pickle.dump(train_data, f)
    
    # Save test data
    test_data = {
        "images": test_images,
        "labels": test_labels,
        "num_classes": ds["test"].features["label"].num_classes
    }
    
    with open(data_dir / "test_preprocessed.pkl", "wb") as f:
        pickle.dump(test_data, f)
    
    # Save dataset info
    dataset_info = {
        "num_classes": ds["train"].features["label"].num_classes,
        "train_size": len(train_images),
        "test_size": len(test_images),
        "image_shape": train_images[0].shape,
        "class_names": ds["train"].features["label"].names
    }
    
    with open(data_dir / "dataset_info.pkl", "wb") as f:
        pickle.dump(dataset_info, f)
    
    print("Preprocessing complete!")
    print(f"Train set: {len(train_images)} samples")
    print(f"Test set: {len(test_images)} samples")
    print(f"Number of classes: {dataset_info['num_classes']}")
    print(f"Image shape: {dataset_info['image_shape']}")
    print(f"Data saved in: {data_dir}")

if __name__ == "__main__":
    preprocess_and_save_dataset()