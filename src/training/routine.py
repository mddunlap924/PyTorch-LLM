import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from utils.data.load_data import LoadData
from utils.data.load_datasets import TrainDataset, TestDataset, collate, collate_fn
from utils.data import splitters
from utils import nlp
from utils.training import assess_epoch
import wandb

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from models import custom_models

def workflow(cfg):

    
    # Load Data
    csv_load_data = LoadData(base_dir='data/')
    train = csv_load_data.load(file_type='train')
    test = csv_load_data.load(file_type='test')

    # Split Dataset
    train = splitters.multilabelstrat(df=train,
                                      n_splits=CFG.num_folds,
                                      target_cols=TARGET_COLS,
                                      seed=CFG.seed)

    # Tokenizer
    tok = nlp.tokenizers.SelectTokenizer(tokenizer_path=CFG.tokenizer.name,
                                         abbreviations=[None]).tokenizer()
    collater = DataCollatorWithPadding(tokenizer=tok, return_tensors='pt')

    # Training for each Fold
    for fold in range(CFG.num_folds):

        # Datasets and dataloaders
        train_dataset = TrainDataset(df=train[train.fold != fold],
                                    tokenizer=tok,
                                    tokenizer_cfg=CFG.tokenizer,
                                    target_cols=TARGET_COLS)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CFG.batch_size,
                                      collate_fn=collater,
                                      shuffle=True,
                                      num_workers=CFG.num_workers)
        val_dataset = TrainDataset(df=train[train.fold == fold],
                                   tokenizer=tok,
                                   tokenizer_cfg=CFG.tokenizer,
                                   target_cols=TARGET_COLS)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=CFG.batch_size,
                                    # collate_fn=collate_fn,
                                    shuffle=True,
                                    num_workers=CFG.num_workers)
        
        
        model = CustomModel(CFG, config_path=None, pretrained=True)
        torch.save(model.config, OUTPUT_DIR+'config.pth')
        model.to(device)
        
        def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if "model" not in n],
                'lr': decoder_lr, 'weight_decay': 0.0}
            ]
            return optimizer_parameters

        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=CFG.encoder_lr, 
                                                    decoder_lr=CFG.decoder_lr,
                                                    weight_decay=CFG.weight_decay)
        optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
        
        

        for epoch in range(CFG.epochs):
            # Train for a single epoch
            train_loss = assess_epoch.train_fn(fold=fold,
                                                train_loader=train_dataloader,
                                                model=model,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                epoch=CFG.epochs)
        for step, inputs in enumerate(train_dataloader):
            inputs = inputs.to(DEVICE)
            input_ids = inputs['input_ids']
            labels = inputs['labels']
            print(step)
            print('check point')

    
    
    return