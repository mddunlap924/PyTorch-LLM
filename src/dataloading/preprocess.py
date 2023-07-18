import numpy as np
from sklearn import preprocessing


class PreprocessData():
    def __init__(self, y: np.array,
                 technique: str):
        # Encoding technique
        if technique == 'LabelEncoder':
            enc = preprocessing.LabelEncoder()
        elif technique == 'OneHotEncoder':
            enc = preprocessing.OneHotEncoder()
        else:
            raise ValueError((f'Encoder needs to be added '
                            f'to script: {technique}'))
        # Fit the encoder
        enc.fit(y)

        # Encoder
        self.encoder = enc
    