""" Implementation of custom pytorch dataset for batched embedding extraction.

"""
import pandas as pd
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """ Custom dataset for batch dialogue embedding extraction. 
    Returns utterance/dialogue ID (based on the dataset) and
    corresponding text data.

    """
    def __init__(self, csv_path: str):
        self.dataframe = pd.read_csv(csv_path, sep=",", encoding="utf-8")

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row["text"]
        utt_id = row["id"]
        return utt_id, text

    def __len__(self):
        return len(self.dataframe)
