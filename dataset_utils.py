"""
Custom dataset class for loading preprocessed data.
"""

import pickle
from pathlib import Path
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    """Dataset class for loading preprocessed data from pickle files."""
    
    def __init__(self, data_file):
        """
        Args:
            data_file (str or Path): Path to the preprocessed data pickle file.
        """
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        
        self.images = data["images"]
        self.labels = data["labels"]
        self.num_classes = data["num_classes"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx]
        }

def load_dataset_info(data_dir="data"):
    """Load dataset information from pickle file."""
    data_dir = Path(data_dir)
    with open(data_dir / "dataset_info.pkl", "rb") as f:
        return pickle.load(f)

def create_data_loaders(data_dir="data", batch_size=512, shuffle_train=True):
    """Create DataLoaders for preprocessed train and test data."""
    data_dir = Path(data_dir)
    
    # Load datasets
    train_dataset = PreprocessedDataset(data_dir / "train_preprocessed.pkl")
    test_dataset = PreprocessedDataset(data_dir / "test_preprocessed.pkl")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader, train_dataset.num_classes