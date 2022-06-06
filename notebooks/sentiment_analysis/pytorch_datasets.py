import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from enum import Enum


class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    # VAL = 3


class SentimentAnalysisDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: AutoTokenizer,
                 max_token_len: int = 512,
                 stratify_column_name: str = "label",
                 frac_train: float = 0.8,
                 frac_test: float = 0.2,
                 random_state = 42):

        self.df = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.stratify_column_name = stratify_column_name
        self.frac_train = frac_train
        self.frac_test = frac_test
        self.random_state = random_state

        # Initialize dataset and labels as None
        self.dataset = None
        self.labels = None

        # Stratified train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = self.stratified_train_test_split()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Set post and label
        post: str = str(self.dataset.iloc[idx])
        label = self.labels.iloc[idx]

        # Tokenize post
        tokens = self.tokenizer.encode_plus(
            post,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'].flatten(),
            'attention_mask': tokens['attention_mask'].flatten(),
            'token_type_ids': tokens["token_type_ids"].flatten(),
            'labels': torch.FloatTensor(label)
        }

    def stratified_train_test_split(self):
        X = self.df # Contains all columns.
        y = self.df[[self.stratify_column_name]] # Dataframe of just the column on which to stratify.

        # Split original dataframe into train and temp dataframes.
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=(1.0 - self.frac_train),random_state=self.random_state)

        return X_train, X_test, y_train, y_test

    def set_fold(self, type: str):
        # It's important to call this method before using the dataset
        if type == DatasetType.TRAIN:
            self.dataset, self.labels = self.X_train, self.y_train
        if type == DatasetType.TEST:
            self.dataset, self.labels = self.X_test, self.y_test
        return self