from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import torch


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


def collate_fn(inputs):
    mask_len = max([int(i[0]['attention_mask'].sum()) for i in inputs])
    for count, input in enumerate(inputs):
        for k, v in input[0].items():
            inputs[count][0][k] = input[0][k][:mask_len]
    return inputs


def prepare_input(tokenizer, cfg, text):
    inputs = tokenizer.encode_plus(
        text=text,
        return_tensors='pt',
        add_special_tokens=cfg.add_special_tokens,
        max_length=cfg.max_length,
        padding=cfg.padding,
        truncation=cfg.truncation,
    )
    # for k, v in inputs.items():
    #     inputs[k] = torch.tensor(v, dtype=torch.long)

    return inputs


class TrainDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tok,
                 tok_cfg,
                 X_cols: list[str],
                 label: str,
                 encoder):
        self.df = df
        self.tokenizer = tok
        self.tokenizer_cfg = tok_cfg
        self.X_cols = X_cols
        self.label = label
        self.encoder = encoder


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        # Extract all source fields into a list
        texts = []
        for col in self.X_cols:
            if col == 'State':
                text = f'State {self.df[col].iloc[idx]}'
            elif col == 'Company response to consumer':
                text = f'Response {self.df[col].iloc[idx]}'
            elif col == 'Consumer complaint narrative':
                text = self.df[col].iloc[idx]
            texts.append(text)

        # Combine the fields using special SEP token
        texts = '[SEP]'.join(texts)

        # Tokenize the text
        inputs = prepare_input(tokenizer=self.tokenizer,
                               cfg=self.tokenizer_cfg,
                               text=texts)

        # Convert labels into one-hot encoded
        labels = self.df[self.label].iloc[idx]
        labels = self.encoder.transform([[labels]])
        labels = torch.tensor(labels, dtype=torch.int)
        return {'inputs': inputs, 'labels': labels}


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, tokenizer_cfg):
        self.tokenizer = tokenizer
        self.tokenizer_cfg = tokenizer_cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(tokenizer=self.tokenizer,
                               cfg=self.tokenizer_cfg,
                               text=self.texts[item])
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.float)
        return {'input_ids': input_ids}
