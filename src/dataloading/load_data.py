from pathlib import Path
import pandas as pd


# Load Data
class LoadData:
    """ 
    Load CSV Data Files
    (Expand this class to other datasets suitable for your needs)
    """

    def __init__(self, base_dir: str):
        """
        :param base_dir: Directory data files are stored
        """

        self.base_dir = Path(base_dir)


    def load(self, filename: str) -> pd.DataFrame:
        """
        Pandas Read CSV filename 
        :param filename: Name of File to Load
        :return: Data returned as a Pandas DataFrame
        """
        return pd.read_csv(self.base_dir / filename,
                           low_memory=False)
