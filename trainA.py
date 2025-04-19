import argparse
import os
import random
import zipfile
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Activation mapping
activation_map = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "Mish": nn.Mish
}

# CNN model
class CNN(nn.Module):
    def __init__(self, filters, activation_fn, batch_norm, dropout):
        super(CNN, self).__init__()
        layers = []
        in_channels = 3
        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_fn())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1] * (224 // (2 ** len(filters))) ** 2, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# Download dataset if not present
def download_dataset():
    if not os.path.exists("inaturalist_12K"):
        print("Downloading dataset...")
        urllib.request.urlretrieve("https://storage.googleapis.com/wandb_datasets/nature_12K.zip", "nature_12K.zip")
        with zipfile.ZipFile("nature_12K.zip", "r") as zip_ref:
            zip_ref.extractall()
        os.remove("nature_12K.zip")

# Load final data
def load_final_data(batch_size, data_aug):
    base_transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    if data_aug:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            *base_transform
        ])
    else:
        transform = transforms.Compose(base_transform)

    dataset = datasets.ImageFolder("inaturalist_12K/train", transform=transform)
    train_idx, val_idx = train_test_split(range(len(dataset)), stratify=[s[1] for s in dataset.samples], test_size=0.2, random_state=42)
    merged_dataset = ConcatDataset([Subset(dataset, train_idx), Subset(dataset, val_idx)])
    train_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = datasets.ImageFolder("inaturalist_12K/val", transform=transforms.Compose(base_transform))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, test_dataset.classes

# Train final model
def train_final_model(args):
    train_loader, test_loader, class_names = load_final_data(args.batch_size, args.data_aug)
    model = CNN(args.filters, activation_map[args.activation], args.batch_norm, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}: Train Acc = {correct / total:.4f} | Loss = {total_loss / len(train_loader):.4f}")

    return model, test_loader, class_names

# Compute test accuracy
def compute_test_accuracy(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f"\nFinal Test Accuracy: {acc:.2f}%")
    wandb.log({"Test Accuracy (%)": acc})

# Show predictions
def show_test_predictions(model, test_loader, class_names):
    model.eval()
    all_images, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())
            all_preds.extend(preds.cpu())

    idxs = random.sample(range(len(all_images)), 30)
    fig, axes = plt.subplots(10, 3, figsize=(12, 40), dpi=150)
    for i, idx in enumerate(idxs):
        row, col = i // 3, i % 3
        img = all_images[idx].permute(1, 2, 0) * 0.5 + 0.5
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        true_label = class_names[all_labels[idx]]
        pred_label = class_names[all_preds[idx]]
        axes[row, col].set_title(f"T: {true_label}\nP: {pred_label}",
                                 fontsize=12,
                                 color='green' if true_label == pred_label else 'red')

    plt.tight_layout()
    plt.savefig("test_predictions_grid.png", bbox_inches='tight')
    wandb.log({"Prediction Grid": wandb.Image("test_predictions_grid.png")})
    plt.show()

# CLI Entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on iNaturalist with wandb")
    parser.add_argument("--entity", type=str, required=True, help="W&B entity")
    parser.add_argument("--project", type=str, default="Assignment_02A", help="W&B project name")
    parser.add_argument("--filters", nargs="+", type=int, default=[64, 128, 256, 128, 64], help="Conv layer filters")
    parser.add_argument("--activation", type=str, choices=activation_map.keys(), default="SiLU", help="Activation function")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch_norm", type=bool, default=True, help="Use BatchNorm")
    parser.add_argument("--data_aug", type=bool, default=False, help="Use data augmentation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    args = parser.parse_args()

    # Ensure dataset is available
    download_dataset()

    # Init wandb
    wandb.init(project=args.project, entity=args.entity, config=vars(args), name="train-run")
    model, test_loader, class_names = train_final_model(args)
    compute_test_accuracy(model, test_loader)
    show_test_predictions(model, test_loader, class_names)
    wandb.finish()
