import pandas as pd
from typing import List
import socket
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def multilabelstrat(df: pd.DataFrame,
                    n_splits: int,
                    target_cols: List[str],
                    *,
                    shuffle: bool=True, seed: int=42) -> pd.DataFrame:
    mlstrat = MultilabelStratifiedKFold(n_splits=n_splits,
                                        shuffle=shuffle,
                                        random_state=seed)
    for n, (_, val_idx) in enumerate(mlstrat.split(df, df[target_cols])):
        df.loc[val_idx, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    return df
