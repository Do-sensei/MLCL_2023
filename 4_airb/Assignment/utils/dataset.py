# utils/dataset.py
import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")

class ReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        # TODO: Implement the constructor
        # self.tokenizer = # FILL_THIS_IN
        self.data = dataframe.dropna(subset=['Review','Positive or Negative'])
        self.data = self.data.reset_index(drop=True)
        
        # TODO: find the review(data) and targets(label) 
        #   from the dataframe using self.data
        # self.review = # FILL_THIS_IN
        self.targets = # FILL_THIS_IN
        
        # TODO: set the max_len
        self.max_len = # FILL_THIS_IN


    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = str(self.review.iloc[idx])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[idx], dtype=torch.float)
        }
