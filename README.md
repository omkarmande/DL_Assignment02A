# CNN Training on iNaturalist Dataset

This project trains a customizable CNN on the [iNaturalist mini dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) using PyTorch and logs metrics using [Weights & Biases (wandb)](https://wandb.ai).

---

## Dataset

The dataset is automatically downloaded and extracted from:
```
https://storage.googleapis.com/wandb_datasets/nature_12K.zip
```

Structure after extraction:
```
inaturalist_12K/
├── train/   # Training images (used with stratified 80-20 split for training and validation)
└── val/     # Test images (used as final test set)
```

---

## Model Architecture

The CNN consists of:
- 5 convolutional blocks: each with Conv2D → (optional) BatchNorm → Activation → MaxPool2D
- 1 fully connected (dense) layer
- 1 output layer with 10 classes

Customizable options include:
- Number of filters per layer
- Activation function (`ReLU`, `GELU`, `SiLU`, `Mish`)
- Dropout rate
- Batch normalization toggle
- Data augmentation toggle

---

## Setup & Installation

1. Install Python packages:
```bash
pip install torch torchvision wandb matplotlib scikit-learn tqdm
```

2. (Optional) Log in to wandb:
```bash
wandb login
```

---

## Usage

Run the training script from command line:

```bash
python train.py \
  --entity <your_wandb_entity> \
  --project Assignment_02A \
  --filters 64 128 256 128 64 \
  --activation SiLU \
  --dropout 0.3 \
  --batch_norm True \
  --data_aug True \
  --batch_size 32 \
  --lr 1e-4 \
  --epochs 10
```

Replace `<your_wandb_entity>` with your wandb username or team.

---

## Logging with wandb

The script logs:
- Training accuracy and loss per epoch
- Final test accuracy
- A grid of 30 test predictions (green = correct, red = incorrect)

You can view all logs and images on your wandb dashboard after training.

---

## Output

- `test_predictions_grid.png`: saved prediction grid showing true vs predicted labels.
- wandb dashboard: interactive plots and metrics for all experiments.

---

## Notes

- The dataset is only downloaded once if not already present.
- Validation split is stratified to ensure class balance (20% of training data).
- The model is trained on GPU if available.

---

## Author

Developed for **DA6401 Assignment 02A** – Fine-tuning and training CNN models on the iNaturalist dataset.
