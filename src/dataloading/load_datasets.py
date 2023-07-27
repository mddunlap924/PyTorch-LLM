from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader


class CustomTextCollator:
    """
    Data Collator used for a classification task. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, tokenizer, tokenizer_cfg):

        # Tokenizer to be used inside the class.
        self.tokenizer = tokenizer

        # Tokenizer configuration
        self.tok_cfg = tokenizer_cfg

        # Check max sequence length.
        self.max_sequence_len = tokenizer_cfg.max_length
        return


    def __call__(self, sequences):
        """
        This function allows the class objects to be used as a function call.
        Since the PyTorch DataLoader needs a collator function, this 
        class can be used as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holds the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]

        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        # https://huggingface.co/docs/transformers/pad_truncation
        inputs = self.tokenizer(text=texts,
                                return_tensors=self.tok_cfg.return_tensors,
                                padding=self.tok_cfg.padding,
                                truncation=self.tok_cfg.truncation,
                                max_length=self.max_sequence_len,
                                add_special_tokens=self.tok_cfg.add_special_tokens,
                                )
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels': torch.tensor(labels, dtype=torch.long)})
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
        text = []
        for col in self.X_cols:
            if col == 'ZIP code':
                feature = f'Zip code {self.df[col].iloc[idx]}'
            elif col == 'Sub-issue':
                feature = f'{self.df[col].iloc[idx]}'
            elif col == 'Consumer complaint narrative':
                feature = self.df[col].iloc[idx]
            text.append(feature)

        # Combine the fields using special SEP token
        text = '[SEP]'.join(text)
        # Extract all source fields into a list
        # text = self.df['Consumer complaint narrative'].iloc[idx]

        # Convert text labels into labels (e.g., if 18 classes then labels are 0-17)
        label_text = self.df[self.label].iloc[idx]
        label = self.encoder.transform([label_text])[0]
        return {'text': text, 'label': label}


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


def get_ds_dl(df,
              cfg,
              tokenizer,
              encoder,
              collator):
    "Get the PyTorch Dataset (ds) and Dataloader (dl)"
    # Dataset 
    ds = TrainDataset(df=df,
                      tok=tokenizer,
                      tok_cfg=cfg.tokenizer,
                      X_cols=cfg.data_info.source_fields,
                      label=cfg.data_info.target,
                      encoder=encoder)

    # Dataloader
    dl = DataLoader(ds,
                    batch_size=cfg.batch_size,
                    collate_fn=collator,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    )
    return ds, dl
