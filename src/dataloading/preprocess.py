import numpy as np
from sklearn.preprocessing import (LabelEncoder,
                                   OneHotEncoder)


class PreprocessData():
    def __init__(self, y: np.array,
                 technique: str):
        # Encoding technique
        if technique == 'LabelEncoder':
            enc = LabelEncoder()
            # Fit the encoder
            enc.fit(y)
        elif technique == 'OneHotEncoder':
            enc = OneHotEncoder(sparse_output=False)
            # Fit the encoder
            y = np.array([[i] for i in np.unique(y).tolist()])
            enc.fit(y)
        else:
            raise ValueError((f'Encoder needs to be added '
                            f'to script: {technique}'))

        # Encoder
        self.encoder = enc
    