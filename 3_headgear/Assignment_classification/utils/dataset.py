# ./utils/dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from utils.config import load_config

import warnings
warnings.filterwarnings("ignore")

class HeadGearDataset(Dataset):
    def __init__(self, annotations_file, dataset_path, mode, transform=None, target_transform=None):
        self.config = load_config('configs/configs.yaml')  # moved here
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[self.img_labels['data set'] == mode]
        
        # TODO: Define the attributes of this dataset
        # self.transform = # fill this in
        # self.target_transform = # fill this in
        # self.dataset_path = # fill this in
        
    def __len__(self):
        # TODO: Return the length of the dataset
        
    def __getitem__(self, idx):
        # TODO: Return the idx-th item of the dataset
        # img_path = # fill this in  # 'filepaths' column
        
        # TODO: path join
        # img_path = # fill this in
        image = self.load_image(img_path)
        
        # TODO: Return the idx-th item of the dataset
        # label = # fill this in # 'class id' column
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def load_image(self, path):
        with Image.open(path) as img:
            img.load()  # This forces the image file to be read into memory
            return img  # Return the PIL image directly
