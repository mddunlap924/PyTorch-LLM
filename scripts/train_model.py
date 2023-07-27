# Libraries
from pathlib import Path
import gc
import argparse
from transformers import AutoTokenizer
import torch

# Append Path to Custom Modules if Needed
# sys.path.append('./')

# Custom Modules
from src.utils import (seed_everything,
                       load_cfg,
                       RunIDs,
                       debugger_is_active,
                       plot_perf_metric_to_disk)
from src.dataloading.load_data import LoadData
from src.dataloading.stratify import StratifyData
from src.dataloading.preprocess import PreprocessData
from src.dataloading.load_datasets import (CustomTextCollator,
                                           get_ds_dl,
                                           )
from src.training.single_fold import train_fold

# Seed Everything
SEED = 42
seed_everything(seed=SEED)

# Get Device type for processing
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def workflow():
    """
    The workflow for training a PyTorch model
    """

    # Load Data from Disk
    load_data_file = LoadData(base_dir=CFG.paths.data.base_dir)
    if CFG.debug:
        data = load_data_file.load(filename=CFG.paths.data.debug_data)
    else:
        data = load_data_file.load(filename=CFG.paths.data.data)

    # Stratify the Data
    data = (StratifyData(technique=CFG.stratify.technique,
                         n_folds=CFG.cv.num_folds,
                         target=CFG.data_info.target)
            .stratify(df=data))

    # Train a model for each validation fold
    for fold_num in CFG.cv.val_folds:
    # fold_num = CFG.cv.val_folds[0]

        print((f'''
            # ====================================================
            # Starting Training for FOLD: {fold_num}
            # ====================================================
            '''))

        # Split Data into Training and Validation
        df_train = data.copy()[data.fold != fold_num].reset_index(drop=True)
        df_val = data.copy()[data.fold == fold_num].reset_index(drop=True)
        print(f'Train Number of Instances: {len(df_train):,}')
        print(f'Validation Number of Instances: {len(df_val):,}')

        # Preprocessing Encoders
        encoders = {}
        for technique in CFG.preprocessing.apply_techniques:
            fields = getattr(CFG.preprocessing, technique).fields
            for col in fields:
                enc = PreprocessData(y=df_train[col].values,
                                    technique=technique)
                encoders[col] = {'encoder': enc.encoder,
                                'technique': technique}

        # Path to the model and tokenizer model card saved on disk
        model_path = Path(CFG.model_tokenizer.base_dir) / CFG.model_tokenizer.name

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower=True)

        # Collator
        collator = CustomTextCollator(tokenizer=tokenizer,
                                    tokenizer_cfg=CFG.tokenizer)

        # Train Dataset and Dataloader
        (_,
        train_dataloader) = get_ds_dl(df=df_train,
                                    cfg=CFG,
                                    tokenizer=tokenizer,
                                    encoder=encoders[CFG.data_info.target]['encoder'],
                                    collator=collator)
        # Validation Dataset and Dataloader
        (_,
        val_dataloader) = get_ds_dl(df=df_val,
                                    cfg=CFG,
                                    tokenizer=tokenizer,
                                    encoder=encoders[CFG.data_info.target]['encoder'],
                                    collator=collator)

        print(f'# of Training Samples: {len(df_train):,}')
        print(f'# of Validation Samples: {len(df_val):,}')
        print(f'Batch Size: {CFG.batch_size}')
        print(f'{len(df_train):,} \ {CFG.batch_size:,} = {len(train_dataloader):,}')
        print(f'Train DataLoader # of Iters: {len(train_dataloader):,}')
        print(f'Val. DataLoader # of Iters: {len(val_dataloader):,}')

        # Path to save model results
        model_save_path = getattr(run_ids.folds_id, f'fold{fold_num}').path

        # Training for a single fold
        perf_metrics = train_fold(train_dl=train_dataloader,
                                val_dl=val_dataloader,
                                cfg=CFG,
                                device=DEVICE,
                                n_classes=df_train[CFG.data_info.target].nunique(),
                                model_save_path=model_save_path,
                                )

        # Save plots of performance metrics to disk for visual assessment
        if CFG.paths.save_results.apply_metric:
            epochs = perf_metrics['epoch']
            for metric_name in ['loss', 'f1', 'precision', 'recall']:
                save_path = model_save_path / f'{metric_name}.png'
                plot_perf_metric_to_disk(save_path=save_path,
                                        x=epochs,
                                        y_train=perf_metrics[f'train_{metric_name}'],
                                        y_val=perf_metrics[f'val_{metric_name}'],
                                        metric_name=metric_name,
                                        )
        print(f'Completed Training Fold {fold_num}\n')

        # Clean up
        del (tokenizer, collator, train_dataloader, val_dataloader,
             model_save_path, perf_metrics, encoders, df_train, df_val)
        _ = gc.collect()
    return


if __name__ == '__main__':

    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = './cfgs'
        args.name = 'train-1-debug.yaml'
    else:
        arg_desc = '''This program points to input parameters for model training'''
        parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                         description= arg_desc)
        parser.add_argument("-cfg_dir",
                            "--dir",
                            required=True,
                            help = "Base Dir. for the YAML config. file")
        parser.add_argument("-cfg_filename",
                            "--name",
                            required=True,
                            help="File name of YAML config. file")
        args = parser.parse_args()
        print(args)

    # Load the configuration file
    CFG = load_cfg(base_dir=Path(args.dir),
                   filename=args.name)

    # Create directories for saving results and use unique Group ID
    run_ids = RunIDs(test_folds=CFG.cv.val_folds,
                     num_folds=CFG.cv.num_folds,
                     save_dir=CFG.paths.save_results.base_dir,
                     save_results=CFG.paths.save_results.apply_metric)
    run_ids.generate_run_ids()

    # Start the training workflow
    workflow()

    print('PYTHON SCRIPT COMPLETED - END')
    