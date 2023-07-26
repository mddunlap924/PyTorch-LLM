# Libraries
import sys
import os
from pathlib import Path
import gc
import argparse
import inspect
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch


# Append Path to Custom Modules
sys.path.append('./')

# Custom Modules
from src.models import llm_multiclass
from src.utils import (RecursiveNamespace,
                       seed_everything,
                       load_cfg,
                       RunIDs,
                       debugger_is_active)
from src.dataloading.load_data import LoadData
from src.dataloading.stratify import StratifyData
from src.dataloading.preprocess import PreprocessData
from src.dataloading.load_datasets import (TrainDataset,
                                           CustomTextCollator,
                                           get_ds_dl,
                                           )
from src.models.llm_multiclass import CustomModel
from src.training.optimizers import get_optimizer
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

    #TODO START LOOPING OVER EACH FOLD HERE
    # Train a model for each validation fold
    fold_num = CFG.cv.val_folds[0]
    print(f'Starting Training for Fold {fold_num}')

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

    # Training for a single fold
    val_metrics = train_fold(train_dl=train_dataloader,
                             val_dl=val_dataloader,
                             cfg=CFG,
                             device=DEVICE,
                             n_classes=df_train[CFG.data_info.target].nunique(),
                             )
    return


if __name__ == '__main__':

    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = './cfgs'
        args.name = 'train-1.yaml'
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
                     save_results=CFG.paths.save_results.apply)
    run_ids.generate_run_ids()

    # Start the training workflow
    workflow()

    # # Load Data from Disk
    # load_data_file = LoadData(base_dir=CFG.paths.data.base_dir)
    # if CFG.debug:
    #     data = load_data_file.load(filename=CFG.paths.data.debug_data)
    # else:
    #     data = load_data_file.load(filename=CFG.paths.data.data)

    # # Stratify the Data
    # data = (StratifyData(technique=CFG.stratify.technique,
    #                      n_folds=CFG.cv.num_folds,
    #                      target=CFG.data_info.target)
    #         .stratify(df=data))
    # cols = CFG.data_info.source_fields + \
    #     [CFG.data_info.target, 'fold']

    # # Number of classes for downstream use
    # N_CLASSES = data[CFG.data_info.target].nunique()

    # from torch.utils.data import DataLoader
    # # Train a model for each validation fold
    # fold_num = CFG.cv.val_folds[0]

    # # Split Data into Training and Validation
    # df_train = data.copy()[data.fold != fold_num].reset_index(drop=True)
    # df_val = data.copy()[data.fold == fold_num].reset_index(drop=True)
    # print(f'Train Number of Instances: {len(df_train):,}')
    # print(f'Validation Number of Instances: {len(df_val):,}')
    
    # # Preprocessing Encoders
    # encoders = {}
    # for technique in CFG.preprocessing.apply_techniques:
    #     fields = getattr(CFG.preprocessing, technique).fields
    #     for col in fields:
    #         enc = PreprocessData(y=df_train[col].values,
    #                             technique=technique)
    #         encoders[col] = {'encoder': enc.encoder,
    #                         'technique': technique}
            
    # from transformers import AutoTokenizer, DataCollatorWithPadding
    # # Path to the model and tokenizer model card saved on disk
    # model_path = Path(CFG.model_tokenizer.base_dir) / CFG.model_tokenizer.name

    # # Load the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower=True)

    # # Collator
    # collator = CustomTextCollator(tokenizer=tokenizer,
    #                             tokenizer_cfg=CFG.tokenizer)
    
    # # Train Dataset and Dataloader
    # train_dataset = TrainDataset(df=df_train,
    #                             tok=tokenizer,
    #                             tok_cfg=CFG.tokenizer,
    #                             X_cols=CFG.data_info.source_fields,
    #                             label=CFG.data_info.target,
    #                             encoder=encoders[CFG.data_info.target]['encoder'])
    # train_dataloader = DataLoader(train_dataset,
    #                             batch_size=CFG.batch_size,
    #                             collate_fn=collator,
    #                             shuffle=True,
    #                             num_workers=CFG.num_workers,
    #                             pin_memory=True,
    #                             )

    # # Validation Dataset and Dataloader
    # val_dataset = TrainDataset(df=df_val,
    #                         tok=tokenizer,
    #                         tok_cfg=CFG.tokenizer,
    #                         X_cols=CFG.data_info.source_fields,
    #                         label=CFG.data_info.target,
    #                         encoder=encoders[CFG.data_info.target]['encoder'])
    # val_dataloader = DataLoader(val_dataset,
    #                             batch_size=CFG.batch_size,
    #                             collate_fn=collator,
    #                             shuffle=True,
    #                             num_workers=CFG.num_workers,
    #                             pin_memory=True,
    #                             )

    # # Total number of steps/iterations
    # total_steps = CFG.epochs * len(train_dataloader)

    # print(f'# of Training Samples: {len(df_train):,}')
    # print(f'# of Validation Samples: {len(df_val):,}')
    # print(f'Batch Size: {CFG.batch_size}')
    # print(f'{len(df_train):,} \ {CFG.batch_size:,} = {len(train_dataloader):,}')
    # print(f'Train DataLoader # of Iters: {len(train_dataloader):,}')
    # print(f'Val. DataLoader # of Iters: {len(val_dataloader):,}')
    
    # from src.models.llm_multiclass import CustomModel
    # # Load custom model
    # model = CustomModel(llm_model_path=model_path,
    #                     cfg=CFG.model,
    #                     num_classes=N_CLASSES)

    # # Set model on device
    # model.to(DEVICE)
    
    # from src.training.optimizers import get_optimizer
    # # Optimizer
    # optimizer = get_optimizer(cfg=CFG.optimizer,
    #                         model=model)
    
    # from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
    # import gc
    # scheduler = CosineAnnealingLR(optimizer,
    #                             T_max=total_steps,
    #                             eta_min=CFG.optimizer.lr.min)
    
    # import time
    # from src.training.metrics import AverageMeter
    # from tqdm import tqdm
    # from torch import nn
    # from torcheval.metrics.functional import (multiclass_f1_score,
    #                                         multiclass_precision,
    #                                         multiclass_recall)
    # from torch.nn.functional import one_hot


    # # Batch size
    # batch_size = CFG.batch_size

    # # Loss Function
    # loss_fn = nn.CrossEntropyLoss()
    # # loss_fn = nn.BCELoss()

    # # # Performance metrics
    # # f1 = MulticlassF1Score(num_classes=N_CLASSES).to(DEVICE)
    # # precision = MulticlassPrecision(num_classes=N_CLASSES).to(DEVICE)
    # # recall = MulticlassRecall(num_classes=N_CLASSES).to(DEVICE)

    # # Training loop over epochs
    # start_training_time = time.time()
    # step_count = 0
    # best_score = 0.0
    # for epoch in range(CFG.epochs):
    #     epoch_start_time = time.time()
    #     print(f'\nStart Epoch {epoch + 1}')
    #     train_meters = {
    #         'loss': AverageMeter(),
    #         'f1': AverageMeter(),
    #         'precision': AverageMeter(),
    #         'recall': AverageMeter(),
    #     }
    #     model.train()
        
    #     # TRAINING
    #     # Iterate over each batch in an epoch
    #     # for idx, batch in enumerate(train_dataloader): [if you don't want a progress bar]
    #     with tqdm(train_dataloader, unit='batch') as tepoch:
    #         for batch in tepoch:
    #             tepoch.set_description(f"Epoch {epoch}")
    #             X = {'input_ids': batch['input_ids'].to(DEVICE),
    #                 'attention_mask': batch['attention_mask'].to(DEVICE)}
    #             y = batch['labels'].to(DEVICE)
    #             # y = one_hot(y, num_classes=N_CLASSES)

    #             # Model prediction
    #             # model.zero_grad()
    #             optimizer.zero_grad()
    #             y_pred_logits = model(X)
    #             y_pred = torch.softmax(y_pred_logits, axis=1)
                
    #             # Calculate loss
    #             loss = loss_fn(input=y_pred, target=y)
                
    #             # Backward pass, optimizer & scheduler steps
    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step()
                
    #             # Performance metrics for the batch of data
    #             f1_score = multiclass_f1_score(y_pred, y, num_classes=N_CLASSES)
    #             precision_score = multiclass_precision(y_pred, y, num_classes=N_CLASSES)
    #             recall_score = multiclass_recall(y_pred, y, num_classes=N_CLASSES)
                
    #             # Store loss and performance metrics
    #             train_meters['loss'].update(loss.detach().cpu().numpy(),
    #                                         n=batch_size)
    #             train_meters['f1'].update(f1_score.detach().cpu().numpy(),
    #                                     n=batch_size)
    #             train_meters['precision'].update(precision_score.detach().cpu().numpy(),
    #                                             n=batch_size)  
    #             train_meters['recall'].update(recall_score.detach().cpu().numpy(),
    #                                         n=batch_size) 

    #             if step_count % 10 == 0:
    #                 tepoch.set_postfix(loss=f'{train_meters["loss"].avg:.4f}',
    #                                 f1=f'{train_meters["f1"].avg:.3f}',
    #                                 precision=f'{train_meters["precision"].avg:.3f}',
    #                                 recall=f'{train_meters["recall"].avg:.3f}')
    #             step_count += 1

    #     # Print training time
    #     print(f'Epoch {epoch + 1} Training Time: '
    #             f'{(((time.time() - epoch_start_time) / 60) / 60):.1f} hrs.')