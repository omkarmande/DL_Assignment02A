import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
INPUT_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataloader with 80/20 split
def get_dataloaders(train_dir, test_dir, batch_size=64):
    full = datasets.ImageFolder(train_dir, transform=transform)
    targets = [label for _, label in full.samples]
    train_idx, val_idx = train_test_split(
        list(range(len(full))), test_size=0.2, stratify=targets, random_state=42
    )
    train_loader = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(Subset(full, val_idx), batch_size=batch_size, shuffle=False, num_workers=2)

    test_ds = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

# Build ResNet50
def build_model(num_classes=10, pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Apply strategy
def apply_strategy(model, strategy, k=None):
    if strategy == "freeze_all_except_last":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    elif strategy == "freeze_first_k":
        children = list(model.children())
        for idx, child in enumerate(children):
            requires_grad = False if idx < k else True
            for p in child.parameters():
                p.requires_grad = requires_grad

    elif strategy == "freeze_last_k":
        children = list(model.children())[:-1]
        total = len(children)
        for idx, child in enumerate(children):
            requires_grad = False if idx >= total - k else True
            for p in child.parameters():
                p.requires_grad = requires_grad
        for p in model.fc.parameters():
            p.requires_grad = True

    elif strategy == "train_from_scratch":
        for p in model.parameters():
            p.requires_grad = True

    return model

# Train one epoch
def train_one_epoch(model, opt, criterion, loader):
    model.train()
    running = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        running += loss.item()
    return running / len(loader)

# Evaluate loop
def evaluate(model, criterion, loader, tag):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            loss_sum += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    wandb.log({f"{tag}_loss": loss_sum / len(loader), f"{tag}_acc": acc})
    print(f"{tag} — loss: {loss_sum / len(loader):.4f}, acc: {acc:.2f}%")

# Main training loop
def main(args):
    train_dl, val_dl, test_dl = get_dataloaders(
        "inaturalist_12K/train", "inaturalist_12K/val", batch_size=64
    )

    wandb.init(entity=args.entity, project=args.project, name=args.strategy, reinit=True)

    print(f"\n=== Strategy: {args.strategy}, k={args.k}, epochs={args.epochs} ===")
    pretrained = False if args.strategy == "train_from_scratch" else True
    model = build_model(pretrained=pretrained).to(device)
    model = apply_strategy(model, args.strategy, args.k)
    model = nn.DataParallel(model)

    optimizer = optim.NAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=0.005
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, criterion, train_dl)
        wandb.log({"epoch": epoch, "train_loss": train_loss})
        evaluate(model, criterion, val_dl, "val")

    evaluate(model, criterion, test_dl, "test")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--strategy", type=str, choices=[
        "freeze_all_except_last", "freeze_first_k", "freeze_last_k", "train_from_scratch"
    ], default="freeze_first_k", help="Training strategy")
    parser.add_argument("--k", type=int, default=4, help="Number of layers to freeze/unfreeze")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()
    main(args)
