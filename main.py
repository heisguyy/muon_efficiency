"""
Main script to compare training with Muon and AdamW optimizers.
"""

import os
import argparse

import torch
from torch import optim
from muon import SingleDeviceMuonWithAuxAdam
from tqdm import tqdm

import wandb
from model import ViTForImageClassification, ViTConfig
from dataset_utils import create_data_loaders

# pylint: disable=invalid-name

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 3e-4

# Load preprocessed data
train_loader, val_loader, num_classes = create_data_loaders(batch_size=BATCH_SIZE)

config = ViTConfig(
    hidden_dropout_prob=0.1,
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=8,
    num_hidden_layers=8,
    num_labels=num_classes,
)
model = ViTForImageClassification(config).to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()

if __name__ == "__main__":
    # Argument parser for optimizer choice
    parser = argparse.ArgumentParser(description="Train ViT with different optimizers")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["muon", "adamw"],
        default="adamw",
        help="Optimizer to use",
    )
    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)
    wandb.init(project="muon_v_adamw", name=f"vit_{args.optimizer}")

    # Initialize optimizer based on user choice
    match args.optimizer:
        case "muon":
            hidden_weights = [p for p in model.vit.encoder.parameters() if p.ndim >= 2]
            hidden_gains_biases = [
                p for p in model.vit.encoder.parameters() if p.ndim < 2
            ]
            nonhidden_params = [
                *model.classifier.parameters(),
                *model.vit.layernorm.parameters(),
                *model.vit.embeddings.parameters(),
            ]
            param_groups = [
                dict(params=hidden_weights, use_muon=True, lr= 50 * LEARNING_RATE),
                dict(
                    params=hidden_gains_biases + nonhidden_params,
                    use_muon=False,
                    lr=LEARNING_RATE,
                ),
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        case "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        training_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        training_loss /= len(train_loader)
        training_accuracy = 100.0 * correct / total

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for test_batch in tqdm(val_loader):
                images = test_batch["image"].to(device)
                labels = test_batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100.0 * correct / total

        # Log metrics
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "train_acc": training_accuracy,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
            }
        )

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {training_loss:.4f}, Train Acc: {training_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "models/best_model.pth")

    print("Training completed!")
    wandb.finish()
