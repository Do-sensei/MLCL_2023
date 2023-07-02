# ./utils/dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from utils.config import load_config

import warnings
warnings.filterwarnings("ignore")

# TODO: Create the HeadGearDataset class
class HeadGearDataset(Dataset):
    def __init__(self, annotations_file, dataset_path, mode, transform=None, target_transform=None):
        self.config = load_config('configs/configs.yaml')  # moved here
        # fiil below in
        
        
        
        
        

    def __len__(self):
        # fiil below in
        return 

    def __getitem__(self, idx):
        # fiil below in
        
        
        
        
        
        return

    def load_image(self, path):
        with Image.open(path) as img:
            img.load()  # This forces the image file to be read into memory
            return img  # Return the PIL image directly
