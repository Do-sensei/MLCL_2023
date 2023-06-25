# training.py
import yaml
import torch
import torch.optim as optim
from torch.nn import MSELoss
from utils.model_mlp import MLP
from utils.dataset import GasDataset
from torch.utils.data import DataLoader

# Load the configuration file
with open('./configs/configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

# TODO: Initialize the model, loss, and optimizer

# Create the training and validation datasets and dataloaders
train_dataset = GasDataset(config['paths']['data'], start_year=None, end_year=2017)
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

val_dataset = GasDataset(config['paths']['data'], start_year=2018, end_year=2019)
val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Training
for epoch in range(config['training']['num_epochs']):
    # Training phase
    model.train()
    train_losses = []
    for inputs, labels in train_loader:
        # TODO: Implement the training step here

    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            # TODO: Implement the validation step here

    print(f'Epoch {epoch+1}/{config["training"]["num_epochs"]}, '
          f'Train Loss: {sum(train_losses)/len(train_losses)}, '
          f'Val Loss: {sum(val_losses)/len(val_losses)}')

# TODO: Save the model
