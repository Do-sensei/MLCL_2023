# test.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.dataset import ReviewDataset
from utils.config import load_config

# TODO: import train_test_split from sklearn
# FILL_THIS_IN

# TODO: import precision_score, recall_score, f1_score, 
#   accuracy_score, confusion_matrix from sklearn.metrics
# FILL_THIS_IN

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Load the configuration
config = load_config('configs/configs.yaml')

# TODO: set the seed using the seed_everything function 
#   with the seed from the config
# FILL_THIS_IN

# TODO: load the data using pandas from the path in the config
# data = # FILL_THIS_IN

# TODO: split the data into test sets
# _, temp_data = # FILL_THIS_IN     # test_size: 0.3, random_state: seed
# _, test_data = # FILL_THIS_IN     # test_size: 0.5, random_state: seed
# WARNING: the training and validation sets are not used in this 'test.py'

# Initialize the tokenizer and the model
# tokenizer = # FILL_THIS_IN    # Check HuggingFace's documentation
# model = # FILL_THIS_IN        # Check HuggingFace's documentation

# Get the number of features of the last layer
num_features = model.classifier.in_features

# Add a new layer without modifying the last layer
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_features, num_features//2),
    torch.nn.ReLU(),
    torch.nn.Linear(num_features//2, 1)
)

# TODO: Load the trained model
# FILL_THIS_IN

# TODO: Initialize the dataset and dataloader
# test_dataset = # FILL_THIS_IN
# test_loader = # FILL_THIS_IN

# Evaluation settings
# TODO: set the device
# device = # FILL_THIS_IN

# TODO: Move the model to the correct device and set it to eval mode
# model = model.'# FILL_THIS_IN' # into the correct device
# model.'# FILL_THIS_IN'         # set it to eval mode

all_predictions = []
all_labels = []
for batch in test_loader:
    # TODO: no gradients
    # with '# FILL_THIS_IN':

        # TODO: move the batch each element
        # input_ids = '# FILL_THIS_IN'.to(device)
        # attention_mask = '# FILL_THIS_IN'.to(device)
        # labels = '# FILL_THIS_IN'.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # TODO: calculate the predictions 'outputs.logits' 
    #   using torch.round and torch.sigmoid
    # predictions = # FILL_THIS_IN
    all_predictions.append(predictions.detach().cpu().numpy())
    all_labels.append(labels.detach().cpu().numpy())

all_predictions = np.concatenate(all_predictions)
all_labels = np.concatenate(all_labels)

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
confusion = confusion_matrix(all_labels, all_predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix: \n{confusion}')

# Define the labels name
labels_name = ["Negative", "Positive"]

# Create a confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_name,
    yticklabels=labels_name,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

# Save the plot
print("Saving the confusion matrix plot...")
plt.savefig("confusion_matrix.png")