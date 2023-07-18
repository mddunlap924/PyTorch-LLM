import pandas as pd
from typing import List
import socket
from sklearn.model_selection import StratifiedKFold


class StratifyData():
    def __init__(self,
                 technique: str,
                 n_folds: int,
                 target: str,
                 *,
                 shuffle: bool=True,
                 seed: int=42):
        # Set parameters
        self.technique = technique
        self.n_folds = n_folds
        self.target = target
        self.shuffle = shuffle
        self.seed = seed


    def __stratified_kfold(self, df: pd.DataFrame) -> pd.DataFrame:
        skf = StratifiedKFold(n_splits=self.n_folds,
                              shuffle=self.shuffle,
                              random_state=self.seed)
        for n, (_, val_idx) in enumerate(skf.split(df, df[self.target])):
            df.loc[val_idx, 'fold'] = int(n + 1)
        df['fold'] = df['fold'].astype(int)
        return df


    def stratify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stratify the dataframe
        """
        stratify_tech = getattr(self, f'_{self.__class__.__name__}__{self.technique}')
        return stratify_tech(df=df)
