
import time
from pathlib import Path
from tqdm import tqdm
import pickle
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


def train_fold(train_dl,
               val_dl,
               cfg,
               device,
               n_classes,
               model_save_path):
    """
    Train a model on a single fold of data
    """
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

    # Set a poor best score when starting
    if cfg.eval_metric.name == 'loss':
        best_score = 1.0E6
    else:
        best_score = 0.0
    metrics = {'epoch': [],
               'train_loss': [],
               'train_f1': [],
               'train_precision': [],
               'train_recall': [],
               'val_loss': [],
               'val_f1': [],
               'val_precision': [],
               'val_recall': []}
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

                # Print at every N steps
                if step_count % 10 == 0:
                    # Extract training metrics by step count
                    train_loss = train_meters["loss"].avg
                    train_f1 = train_meters["f1"].avg
                    train_precision = train_meters["precision"].avg
                    train_recall = train_meters["recall"].avg

                    # Print Metrics to progress bar
                    tepoch.set_postfix(loss=f'{train_loss:.4f}',
                                       f1=f'{train_f1:.3f}',
                                       precision=f'{train_precision:.3f}',
                                       recall=f'{train_recall:.3f}')                   
                step_count += 1

        # Print training time and performance metrics
        print(f'Epoch {epoch + 1} Training Time: '
              f'{(time.time() - epoch_start_time) / 60:.2f} minutes')

        # Extract training metrics by step count
        train_loss = train_meters["loss"].avg
        train_f1 = train_meters["f1"].avg
        train_precision = train_meters["precision"].avg
        train_recall = train_meters["recall"].avg

        print((f'\tTraining: loss={train_loss:.4f}; '
               f'f1={train_f1:.3f}; '
               f'precision={train_precision:.3f}; '
               f'recall={train_recall:.3f}'))

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

        # Extract val. metrics by step count
        val_loss = val_meters["loss"].avg
        val_f1 = val_meters["f1"].avg
        val_precision = val_meters["precision"].avg
        val_recall = val_meters["recall"].avg

        # Print Val. Metric Performance
        print((f'\tVal.: loss={val_loss:.4f}; '
               f'f1={val_f1:.3f}; '
               f'precision={val_precision:.3f}; '
               f'recall={val_recall:.3f}'))

        # Save best model to disk
        if cfg.eval_metric.name == 'loss':
            if val_meters[cfg.eval_metric.name].avg < best_score:
                best_score = val_meters[cfg.eval_metric.name].avg
                save_path = model_save_path / f'model_epoch{epoch + 1}.pt'
                if cfg.paths.save_results.apply_model:
                    torch.save(model.state_dict(), save_path)
                    print(f"Epoch {epoch + 1}; Saved best model at: {save_path}")
        else:
            if val_meters[cfg.eval_metric.name].avg > best_score:
                best_score = val_meters[cfg.eval_metric.name].avg
                save_path = model_save_path / f'model_epoch{epoch + 1}.pt'
                if cfg.paths.save_results.apply_model:
                    print(f"Epoch {epoch + 1}; Saved best model at: {save_path}")

        # Reset metrics after each epoch
        f1.reset()
        precision.reset()
        recall.reset()

        # Store epoch training metrics
        metrics['epoch'] += [epoch]
        metrics['train_loss'] += [train_loss]
        metrics['train_f1'] += [train_f1]
        metrics['train_precision'] += [train_precision]
        metrics['train_recall'] += [train_recall]
        metrics['val_loss'] += [val_loss]
        metrics['val_f1'] += [val_f1]
        metrics['val_precision'] += [val_precision]
        metrics['val_recall'] += [val_recall]

    # Total training time
    total_training_time = (time.time() - start_training_time) / 60
    print(f'Total Training Time: {total_training_time:.1f} minutes')

    # Save performance metrics to disk
    if cfg.paths.save_results.apply_metric:
        with open(model_save_path / f'performance_metrics.pickle', 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return metrics
