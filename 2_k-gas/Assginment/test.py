# test.py
import yaml
import torch
from torch.nn import MSELoss
from utils.model_mlp import MLP
from utils.dataset import GasDataset
from torch.utils.data import DataLoader

# Load the configuration file
with open('./configs/configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

# TODO: Initialize the model and load the saved model weights

# Create the test dataset and dataloader
test_dataset = GasDataset(config['paths']['data'], start_year=2020, end_year=2020)
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Testing
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        # TODO: Implement the testing step here

print(f'Test Loss: {test_loss.item()}')