from pathlib import Path
import pandas as pd


# Load Data
class LoadData:
    """ Load data in CSV files that were supplied from Kaggle  """

    def __init__(self, base_dir: str):
        """
        :param base_dir: Directory CSV files are stored
        """

        self.base_dir = Path(base_dir)
        self.train = self.base_dir / 'train.csv'
        self.test = self.base_dir / 'test.csv'
        self.sub = self.base_dir / 'sample_submission.csv'

    def load(self, file_type: str) -> pd.DataFrame:
        """
        Load CSV file supplied by Kaggle
        :param file_type: Name of File to Load
        :return: Data returned as a Pandas DataFrame
        """
        try:
            if file_type == 'train':
                data = pd.read_csv(self.train)
            elif file_type == 'test':
                data = pd.read_csv(self.test)
            elif file_type == 'sub':
                data = pd.read_csv(self.sub)
            else:
                raise ValueError(f'File Type Does Not Exist "{file_type}"')
            return data
        except ValueError as ve:
            print(f'Error Inside "LoadData.load()": {ve}')
