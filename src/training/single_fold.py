
import time
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      OneCycleLR)
from torcheval.metrics.functional import (multiclass_f1_score,
                                          multiclass_precision,
                                          multiclass_recall)
from torchmetrics.classification import (MulticlassF1Score,
                                         MulticlassPrecision,
                                         MulticlassRecall)
from src.training.metrics import AverageMeter
from src.models.llm_multiclass import CustomModel
from src.training.optimizers import get_optimizer

# ====================================================
# loader
# ====================================================


def train_fold(train_dl,
               val_dl,
               cfg,
               device,
               n_classes):

    # Model path
    model_path = Path(cfg.model_tokenizer.base_dir) / cfg.model_tokenizer.name

    # Load custom model
    model = CustomModel(llm_model_path=model_path,
                        cfg=cfg.model,
                        num_classes=n_classes)
    # Set model on device
    model.to(device)

    # Optimizer
    optimizer = get_optimizer(cfg=cfg.optimizer,
                              model=model)

    # Total number of steps/iterations
    total_steps = cfg.epochs * len(train_dl)

    # Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=total_steps,
                                  eta_min=cfg.optimizer.lr.min)

    # Batch size
    batch_size = cfg.batch_size

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Performance metrics
    f1 = MulticlassF1Score(num_classes=n_classes).to(device)
    precision = MulticlassPrecision(num_classes=n_classes).to(device)
    recall = MulticlassRecall(num_classes=n_classes).to(device)

    # ====================================================
    # Model Training
    # ====================================================
    start_training_time = time.time()
    step_count = 0
    best_score = 0.0
    for epoch in range(cfg.epochs):
        epoch_start_time = time.time()
        print(f'\nStart Epoch {epoch + 1}')
        train_meters = {'loss': AverageMeter(),
                        'f1': AverageMeter(),
                        'precision': AverageMeter(),
                        'recall': AverageMeter()}
        model.train()

        # TRAINING
        # Iterate over each batch in an epoch
        # for idx, batch in enumerate(train_dataloader): [if you don't want a progress bar]
        with tqdm(train_dl, unit='batch') as tepoch:
            for idx, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch + 1}")
                X = {'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)}
                y = batch['labels'].to(device)

                # Model prediction
                optimizer.zero_grad()
                y_pred_logits = model(X)
                y_pred = nn.Softmax(dim=1)(y_pred_logits).argmax(1)

                # Calculate loss
                loss = loss_fn(input=y_pred_logits, target=y)

                # Backward pass, optimizer & scheduler steps
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Performance metrics for the batch of data
                f1_score = f1(y_pred, y)
                precision_score = precision(y_pred, y)
                recall_score = recall(y_pred, y)

                # Store loss and performance metrics
                train_meters['loss'].update(loss.detach().cpu().numpy(),
                                            n=batch_size)
                train_meters['f1'].update(f1_score.detach().cpu().numpy(),
                                          n=batch_size)
                train_meters['precision'].update(precision_score.detach().cpu().numpy(),
                                                 n=batch_size)  
                train_meters['recall'].update(recall_score.detach().cpu().numpy(),
                                              n=batch_size) 

                if step_count % 10 == 0:
                    tepoch.set_postfix(loss=f'{train_meters["loss"].avg:.4f}',
                                       f1=f'{train_meters["f1"].avg:.3f}',
                                       precision=f'{train_meters["precision"].avg:.3f}',
                                       recall=f'{train_meters["recall"].avg:.3f}')
                step_count += 1

        # Print training time and performance metrics
        print(f'Epoch {epoch + 1} Training Time: '
                f'{(((time.time() - epoch_start_time) / 60) / 60):.1f} hrs.')
        print((f'\tTraining: loss={train_meters["loss"].avg:.4f}; '
               f'f1={train_meters["f1"].avg:.3f}; '
               f'precision={train_meters["precision"].avg:.3f}; '
               f'recall={train_meters["recall"].avg:.3f}'))

        # Reset metrics after each epoch
        f1.reset()
        precision.reset()
        recall.reset()

        # ====================================================
        # Evaluate Val. Data After Epoch
        # ====================================================
        val_meters = {'loss': AverageMeter(),
                      'f1': AverageMeter(),
                      'precision': AverageMeter(),
                      'recall': AverageMeter()}
        model.eval()
        with torch.no_grad():
            with tqdm(val_dl, unit='batch') as tepoch:
                for idx, batch in enumerate(tepoch):
                    tepoch.set_description(f"Val. at Epoch: {epoch + 1}")
                    X = {'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device)}
                    y = batch['labels'].to(device)
                    y_pred_logits = model(X)
                    y_pred = nn.Softmax(dim=1)(y_pred_logits).argmax(1)

                    # Calculate loss
                    loss = loss_fn(input=y_pred_logits, target=y)

                    # Performance metrics for the batch of data
                    f1_score = f1(y_pred, y)
                    precision_score = precision(y_pred, y)
                    recall_score = recall(y_pred, y)

                    # Store loss and performance metrics
                    val_meters['loss'].update(loss.detach().cpu().numpy(),
                                              n=batch_size)
                    val_meters['f1'].update(f1_score.detach().cpu().numpy(),
                                            n=batch_size)
                    val_meters['precision'].update(precision_score.detach().cpu().numpy(),
                                                   n=batch_size)  
                    val_meters['recall'].update(recall_score.detach().cpu().numpy(),
                                                n=batch_size) 

        # Print Val. Metric Performance
        print((f'\tVal.: loss={val_meters["loss"].avg:.4f}; '
               f'f1={val_meters["f1"].avg:.3f}; '
               f'precision={val_meters["precision"].avg:.3f}; '
               f'recall={val_meters["recall"].avg:.3f}'))

        # Save best model to disk
        if val_meters["f1"].avg > best_score:
            print(f"Epoch {epoch + 1}; Saved best model at: {'UPDATE ME'}")
            best_score = val_meters['f1'].avg
            # torch.save(model.state_dict(), model_save_path)
        # Reset metrics after each epoch
        f1.reset()
        precision.reset()
        recall.reset()

    # Total training time
    total_training_time = (((time.time() - start_training_time) / 60) / 60)
    print(f'Total Training Time: {total_training_time:.1f} hrs')
    wandb.log({'total_train_time': total_training_time})
    return 'dog'
