"""
Miscellaneous and helper code for various tasks will be used in this script.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import uuid
import random
import yaml
import numpy as np
import torch


class RecursiveNamespace(SimpleNamespace):
    """
    Extending SimpleNamespace for Nested Dictionaries
    # https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8

    Args:
        SimpleNamespace (_type_): Base class is SimpleNamespace

    Returns:
        _type_: A simple class for nested dictionaries
    """
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


def load_cfg(base_dir: Path,
             filename: str,
             *,
             as_namespace: bool=True) -> SimpleNamespace:
    """
    Load YAML configuration files saved uding the "cfgs" directory
    Args:
        base_dir (Path): Directory to YAML config. file
        filename (str): Name of YAML configuration file to load
    Returns:
        SimpleNamespace: A simple class for calling configuration parameters
    """
    cfg_path = Path(base_dir) / filename
    with open(cfg_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
    file.close()
    if as_namespace:
        cfg = RecursiveNamespace(**cfg_dict)
    else:
        cfg = cfg_dict
    return cfg


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def seed_everything(*, seed: int=42):
    """
    Seed everything

    Args:
        seed (_type_): Seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class RunIDs():
    def __init__(self,
                 test_folds: list,
                 num_folds: int):
        self.test_folds = test_folds
        self.num_folds = num_folds
        self.group_id = str
        self.folds_id = RecursiveNamespace

        # Are all folds being tested
        if test_folds == list(range(num_folds)):
            self.test_all_folds = True
        else:
            self.test_all_folds = False


    def generate_id(self):
        # Generate a random ID
        return str(uuid.uuid4()).split('-')[0]


    def generate_run_ids(self):
        # Get a group id (i.e. ID that will organize all folds)
        self.group_id = self.generate_id()

        fold_info = {}
        for fold in self.test_folds:
            fold_info[f'fold{fold}'] = {'run_id': self.generate_id()}
        self.folds_id = RecursiveNamespace(**fold_info)
        return



# class WandB:
#     """ Log results for Weights & Biases """

#     def __init__(self, cfg, group_id):
#         del cfg.checkpoint, cfg.tokenizer
#         self.cfg = cfg
#         self.group_id = group_id
#         self.job_type = wandb.util.generate_id()
#         self.version = wandb.util.generate_id()
#         self.run_name = f'{self.group_id}_{self.job_type}'
#         self.id = wandb.util.generate_id()  # Generate version name for tracking in wandb
#         # WandB Key in the environment variables
#         if 'WANDB_KEY' in os.environ:
#             self.key = os.environ['WANDB_KEY']
#         else:
#             self.key = None

#     def login_to_wandb(self):
#         # Store WandB key in the environment variables
#         if self.key is not None:
#             wandb.login(key=self.key)
#         else:
#             print('Not logging info in WandB')

#     def get_logger(self):
#         self.login_to_wandb()
#         wb_logger = pl_loggers.WandbLogger(project=self.cfg.wandb_project,
#                                             group=self.group_id,
#                                             job_type=self.job_type,
#                                             name=self.run_name,
#                                             version=self.version,
#                                             config=self.cfg,
#                                             )

#         return wb_logger



