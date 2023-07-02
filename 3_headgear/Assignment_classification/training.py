# training.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler

from PIL import Image
from utils.dataset import HeadGearDataset
from utils.resnet_50 import resnet50
from utils.config import load_config
from sklearn.metrics import f1_score

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def train_one_epoch(model, criterion, optimizer, dataloader, device, grad_clip):
    # TODO: Set the model to train mode
    # model.'# fill this in'

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    all_labels = []
    all_predictions = []

    for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        # TODO: Define Output
        # output = # fill this in

        # TODO: Define Loss
        # loss = # fill this in
        running_loss += loss.item() * data.size(0)

        _, predicted = torch.max(output.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        all_labels.extend(target.detach().cpu().numpy().tolist())
        all_predictions.extend(predicted.detach().cpu().numpy().tolist())

        # TODO: Backpropagate Loss
        # loss.'# fill this in'
        
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)
        
        # TODO: Update the weights
        # optimizer.'# fill this in'

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = (correct_predictions / total_predictions) * 100.0
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')

    return epoch_loss, epoch_acc, epoch_f1


def validate(model, criterion, dataloader, device):
    # TODO: Set the model to evaluation mode
    # model.'# fill this in'

    running_valid_loss = 0.0
    total_valid_predictions = 0.0
    correct_valid_predictions = 0.0
    all_valid_labels = []
    all_valid_predictions = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # TODO: Define Output
            # output = # fill this in

            # TODO: Define Loss
            # loss = # fill this in
            running_valid_loss += loss.item() * data.size(0)

            _, predicted = torch.max(output.data, 1)
            total_valid_predictions += target.size(0)
            correct_valid_predictions += (predicted == target).sum().item()

            all_valid_labels.extend(target.detach().cpu().numpy().tolist())
            all_valid_predictions.extend(predicted.detach().cpu().numpy().tolist())

    valid_loss = running_valid_loss / len(dataloader.dataset)
    valid_acc = (correct_valid_predictions / total_valid_predictions) * 100.0
    valid_f1 = f1_score(all_valid_labels, all_valid_predictions, average='macro')

    return valid_loss, valid_acc, valid_f1


def training(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.CenterCrop(64),
        transforms.GaussianBlur(5),
        transforms.ToTensor(),
    ])

    # TODO: Load the train dataset using HeadGearDataset class
    # train_data = # fill this in
    # train_loader = # fill this in

    # TODO: Load the validation dataset and create a DataLoader for it
    # valid_data = # fill this in
    # valid_loader = # fill this in
    
    # TODO: Define the model, loss function, and optimizer
    # model = # fill this in
    model = model.to(device)
    
    # TODO: Define the loss function and optimizer
    # criterion = # fill this in
    criterion = criterion.to(device)
    # optimizer = # fill this in
    
    
    # TODO: Train the model on the training data, and validate it on the validation data
    
    for epoch in range(config['training']['num_epochs']):
        # TODO: Train the model on the training data
        # train_loss, train_acc, train_f1 = train_one_epoch(# fill this in)
        print(f"Epoch: {epoch+1}/{config['training']['num_epochs']}.. Training Loss: {train_loss:.4f}.. Training Accuracy: {train_acc:.2f}%.. Training F1 Score: {train_f1:.2f}")

        # TODO: Validate the model on the validation data
        # valid_loss, valid_acc, valid_f1 = validate(# fill this in)
        print(f"Epoch: {epoch+1}/{config['training']['num_epochs']}.. Validation Loss: {valid_loss:.4f}.. Validation Accuracy: {valid_acc:.2f}%.. Validation F1 Score: {valid_f1:.2f}")


    # TODO: Save the trained model
    # # fill this in
    
if __name__ == "__main__":
    config = load_config('configs/configs.yaml')  # specify the path to your config file
    training(config)
