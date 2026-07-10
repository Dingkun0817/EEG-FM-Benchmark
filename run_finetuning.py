"""
EEG Benchmark: fine-tuning and evaluation entrypoint.

Supports multiple datasets, task modes (Cross / Fewshot), and both deep-learning
and traditional ML models. Run with: python run_finetuning.py --dataset <name> --model_name <name> ...
"""
import os
import json
import copy
import gc
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from sklearn.model_selection import KFold

from utils.utils import split_dataset_cross, create_dataloaders, split_dataset_fewshot
from utils.logger import Logger, get_logger, set_global_logger
from utils.trainer import train_model, evaluate_model
from utils.model_loader import import_dataset, import_model, import_preprocessor, apply_preprocessor
from utils.metrics import get_metrics
from models.config import MODEL_CATEGORY

# Traditional ML model names; derived from config (category == 'ML').
TRADITIONAL_ML_MODELS = tuple(m for m, c in MODEL_CATEGORY.items() if c == 'ML')


def build_model_import_kwargs(args):
    """Build kwargs for ``import_model``; ``readout`` is only passed when set on CLI."""
    kwargs = {'finetune_strategy': args.finetune_strategy, 'dropout_rate': args.dropout_rate}
    if getattr(args, 'readout', None) is not None:
        kwargs['readout'] = args.readout
    return kwargs


# --- CLI / task config ---
def create_parser():
    parser = argparse.ArgumentParser(description='Run fine-tuning for EEG models')
    # Basic config: dataset and model
    parser.add_argument('--dataset', 
                        default='BNCI2014004', 
                        type=str, 
                        help='Dataset name: BNCI2014001 BNCI2014004 BNCI2015001 BNCI2014008 BNCI2014009 CHB_MIT TUAB SleepEDF SEED SEED_VIG Dial ThingsEEG2 EEGMAT')
    
    parser.add_argument('--model_name', 
                        default='CBraMod', 
                        type=str, 
                        help='Model name: ' + ' '.join(MODEL_CATEGORY.keys()))
    # Random seed and GPU
    parser.add_argument('--seeds', 
                        default=[0,1,2], 
                        nargs='+', 
                        type=int, 
                        help='Random seeds for reproducibility')
    
    parser.add_argument('--gpuid', 
                        default=0, 
                        type=int, 
                        help='GPU device ID to use for training')
    # Task mode
    parser.add_argument('--task_mode', 
                        default='Cross', 
                        type=str, 
                        choices=['Cross', 'Fewshot'], 
                        help='Task mode: Cross (normal cross-validation) or Fewshot (use 30%% of each class per subject for training)')
    
    parser.add_argument('--train_percentage', 
                        default=0.3, 
                        type=float,
                        help='Percentage of each class per subject to use for training in few-shot mode (default: 0.3)')

    parser.add_argument('--finetune_strategy',
                        default='full',
                        type=str,
                        choices=['full', 'head_only'],
                        help='Fine-tuning strategy: full (all parameters trainable) or head_only (only task head trainable)')
    # Training
    parser.add_argument('--batch_size', 
                        default=16, 
                        type=int, 
                        help='Batch size for training and evaluation')
    
    parser.add_argument('--dataloader_workers', '--num_workers',
                        default=0,
                        type=int,
                        dest='dataloader_workers',
                        help='Number of worker processes for DataLoader (batch prefetch during training); 0 = main process only')

    parser.add_argument('--data_workers',
                        default=1,
                        type=int,
                        help='Subprocesses for dataset load and preprocessing (filter, resample, etc.); 1 = no parallelism')

    parser.add_argument('--epochs', 
                        default=20, 
                        type=int, 
                        help='Number of training epochs')
    
    parser.add_argument('--dropout_rate', 
                        default=0.5, 
                        type=float, 
                        help='Dropout rate for model task head')

    parser.add_argument('--readout',
                        default=None,
                        type=str,
                        choices=['flatten', 'pooling'],
                        help="FM readout override: 'flatten' or 'pooling'. Omit (default) to use each model loader's default.")
    
    parser.add_argument('--label_smoothing', 
                        default=0, 
                        type=float, 
                        help='Label smoothing parameter for CrossEntropyLoss (default: 0.0, no smoothing)')
    
    parser.add_argument('--class_weights', 
                        default=False, 
                        type=bool, 
                        help='Whether to automatically calculate class weights based on training data label distribution, default False means no weights')
    # Optimizer
    parser.add_argument('--optimizer_type',
                        default='adamw',
                        type=str,
                        help='Optimizer type (e.g., adamw)')
    
    parser.add_argument('--lr', 
                        default=0.001, 
                        type=float, 
                        help='Learning rate')

    parser.add_argument('--weight_decay', 
                        default=0.01, 
                        type=float, 
                        help='Weight decay for optimizer')
    
    parser.add_argument('--opt_eps', 
                        default=1e-8, 
                        type=float, 
                        metavar='EPSILON', 
                        help='Optimizer epsilon (Adam/AdamW eps, default: 1e-8)')
    
    parser.add_argument('--momentum', 
                        type=float, 
                        default=0.9, 
                        metavar='M', 
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--filter_bias_and_bn',
                        default=False,
                        type=bool,
                        help='Whether to filter bias and batch norm parameters from weight decay')
    
    parser.add_argument('--layer_decay',
                        default=1,
                        type=float,
                        help='Layer decay value for hierarchical learning rate scaling (1.0 means no decay)')
    # LR scheduler
    parser.add_argument('--use_lr_scheduler',
                        default=True,
                        type=bool,
                        help='Whether to use learning rate scheduler')
    
    parser.add_argument('--warmup_epochs',
                        default=5,
                        type=int,
                        help='Number of warmup epochs for learning rate scheduler')
    
    parser.add_argument('--min_lr',
                        default=1e-06,
                        type=float,
                        help='Lower lr bound for cyclic schedulers')
    # Weight decay schedule
    parser.add_argument('--use_weight_decay_schedule',
                        default=False,
                        type=bool,
                        help='Whether to use weight decay scheduling')

    parser.add_argument('--min_weight_decay',
                        default=0.0001,
                        type=float,
                        help='Minimum weight decay value when using weight decay scheduling')
    # Gradient clipping
    parser.add_argument('--use_grad_clipping',
                        default=False,
                        type=bool,
                        help='Whether to use gradient clipping')
    
    parser.add_argument('--clip_grad_norm',
                        default=1.0,
                        type=float,
                        help='Maximum norm for gradient clipping (None means no clipping)')
    # Preprocessing
    parser.add_argument('--use_preprocessing_params', 
                        default=True, 
                        type=bool,
                        help='Whether to use preprocessing parameters from parser or default values in preprocessing functions')
    
    parser.add_argument('--target_fs', 
                        default=200, 
                        type=int,
                        help='Target sampling rate for preprocessing')
    
    parser.add_argument('--l_freq', 
                        default=0.3, 
                        type=float,
                        help='Low-pass frequency for filtering')
    
    parser.add_argument('--h_freq', 
                        default=75.0, 
                        type=float,
                        help='High-pass frequency for filtering')
    
    parser.add_argument('--notch_freq', 
                        default=60.0, 
                        type=float,
                        help='Notch filter frequency to remove power line noise')
    
    parser.add_argument('--norm_method', 
                        default=None, 
                        type=str,
                        help='Normalization method (e.g., "0.1mv")')

    parser.add_argument('--time_length', 
                        default=5.0, 
                        type=float,
                        help='Time length for preprocessing')
    
    parser.add_argument('--apply_EA', 
                        default=False, 
                        type=bool,
                        help='Whether to apply EA (Event-related Augmentation) during preprocessing')
    
    return parser


def is_traditional_model(model_name):
    """Return True if model is traditional ML (non–deep learning), for choosing train/eval branch."""
    return model_name in TRADITIONAL_ML_MODELS


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def log_dataset_info(logger, eeg_data, dataset_name, task_type, is_binary, split_method):
    """Log dataset summary to logger."""
    logger.log(f'Dataset: {dataset_name}')
    logger.log(f'Data shape: {eeg_data.eeg_data.shape}')
    logger.log(f'Sampling rate: {eeg_data.sampling_rate} Hz')
    logger.log(f'Task type: {task_type}')
    logger.log(f'Binary: {is_binary}')
    logger.log(f'Split method: {split_method}')


# --- Single-split helpers (used by process_all_splits_dl / process_all_splits_traditional_method) ---
def extract_training_metrics(train_log, logger):
    """Extract final training metrics from train_log."""
    final_train_metrics = {}
    if isinstance(train_log, dict) and 'metrics' in train_log and train_log['metrics'] and isinstance(train_log['metrics'], list):
        last_epoch_metrics = train_log['metrics'][-1]
        if isinstance(last_epoch_metrics, dict):
            for key, value in last_epoch_metrics.items():
                final_train_metrics[key] = value
            if final_train_metrics:
                logger.log(f"Collected training metrics: {', '.join(final_train_metrics.keys())}")
    if not final_train_metrics:
        logger.log("Warning: No training metrics found")
    return final_train_metrics


def extract_all_candidate_features(model, eeg_data, device):
    """Extract all candidate features for matching task; returns img_features or None."""
    if hasattr(eeg_data, 'img_feature'):
        img_features = eeg_data.img_feature
        if isinstance(img_features, np.ndarray):
            img_features = torch.tensor(img_features, dtype=torch.float32).to(device)
        elif hasattr(img_features, 'to'):
            img_features = img_features.to(device)
        img_features = torch.nn.functional.normalize(img_features, dim=1)
        return img_features
    return None


def collect_metrics(seed_metrics, final_metrics, final_train_metrics):
    """Collect test and train metrics into seed_metrics (modified in place)."""
    for metric_key, metric_value in final_metrics.items():
        if metric_key not in seed_metrics:
            seed_metrics[metric_key] = []
        seed_metrics[metric_key].append(metric_value)
    if final_train_metrics:
        for metric_key, metric_value in final_train_metrics.items():
            train_key = f'train_{metric_key}'
            if train_key not in seed_metrics:
                seed_metrics[train_key] = []
            seed_metrics[train_key].append(metric_value)


# --- Multi-split: deep learning vs traditional ML ---
def process_all_splits_dl(splits, model, eeg_data, task_type, is_binary, device, args, logger):
    """Run DL training and evaluation for all splits. Model is discarded after branch check; each split creates a new model via import_model."""
    split_results = []
    seed_losses = []
    seed_metrics = {}
    del model

    for split_idx, split in enumerate(splits):
        logger.log(f'\n\n===== Processing split {split_idx+1}/{len(splits)} =====')
        if 'test_subject' in split:
            logger.log(f'Test subject: {split["test_subject"]}')
        elif 'fold_idx' in split:
            logger.log(f'Fold index: {split["fold_idx"]}')
        train_loader, test_loader = create_dataloaders(
            eeg_data, split['train_mask'], split['test_mask'],
            args.batch_size, device, num_workers=args.dataloader_workers
        )
        if train_loader is None or test_loader is None:
            logger.log(f'Skipping split {split_idx+1} due to data loading error')
            continue

        logger.log(f'\nStarting training for split {split_idx+1}')
        print(f'Using model: {args.model_name}')
        if args.model_name in ('BrainOmni_Tiny', 'BrainOmni_Base'):
            kwargs = build_model_import_kwargs(args)
            model_copy = import_model(args.model_name, eeg_data, **kwargs)
            model_copy = model_copy.to(device)
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            kwargs = build_model_import_kwargs(args)
            model_copy = import_model(args.model_name, eeg_data, **kwargs)
            model_copy = model_copy.to(device)

        k_values = None
        topn_values = None
        if task_type == 'matching':
            k_values = [2, 10]
            topn_values = [1, 5]
        try:
            trained_model, best_val_acc, train_log = train_model(
                model_copy, train_loader, test_loader, task_type, device, args,
                logger=logger, k_values=k_values, topn_values=topn_values
            )
            logger.log(f'\nFinal evaluation on test set for split {split_idx+1}:')
            if task_type == 'regression':
                criterion = nn.MSELoss()
            elif task_type == 'matching':
                from utils.trainer import ClipLoss
                criterion = ClipLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            final_metrics = evaluate_model(
                trained_model, test_loader, criterion, task_type, is_binary=is_binary,
                device=device, logger=logger, k_values=k_values, topn_values=topn_values
            )
            final_loss = final_metrics.get('loss', 0.0)
            final_train_metrics = extract_training_metrics(train_log, logger)
        finally:
            logger.log(f'Cleaning up resources for split {split_idx+1}...')
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / 1024**2
                reserved_before = torch.cuda.memory_reserved() / 1024**2
                logger.log(f'GPU Memory before cleanup - Allocated: {allocated_before:.2f} MB, Reserved: {reserved_before:.2f} MB')
                try:
                    if 'model_copy' in locals():
                        model_copy_size = sum(p.numel() * p.element_size() for p in model_copy.parameters()) / 1024**2
                        logger.log(f'model_copy parameter size: {model_copy_size:.2f} MB')
                    if 'trained_model' in locals():
                        trained_model_size = sum(p.numel() * p.element_size() for p in trained_model.parameters()) / 1024**2
                        logger.log(f'trained_model parameter size: {trained_model_size:.2f} MB')
                        if hasattr(trained_model, 'optimizer'):
                            optimizer_size = sum(buf.numel() * buf.element_size() for buf in trained_model.optimizer.state.values()) / 1024**2
                            logger.log(f'Optimizer state size: {optimizer_size:.2f} MB')
                except Exception as e:
                    logger.log(f'Could not calculate model memory usage: {e}')
            if 'model_copy' in locals():
                del model_copy
            if 'trained_model' in locals():
                del trained_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if 'train_loader' in locals():
                train_loader._iterator = None
                del train_loader
            if 'test_loader' in locals():
                test_loader._iterator = None
                del test_loader
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated() / 1024**2
                reserved_after = torch.cuda.memory_reserved() / 1024**2
                freed_allocated = allocated_before - allocated_after
                freed_reserved = reserved_before - reserved_after
                logger.log(f'GPU Memory after cleanup - Allocated: {allocated_after:.2f} MB, Reserved: {reserved_after:.2f} MB')
                logger.log(f'Memory freed - Allocated: {freed_allocated:.2f} MB, Reserved: {freed_reserved:.2f} MB')
            logger.log(f'Resources cleaned up for split {split_idx+1}')

        split_result = {
            'split_idx': split_idx,
            'test_subject': split.get('test_subject', None),
            'test_subjects': split.get('test_subjects', None),
            'fold_idx': split.get('fold_idx', None),
            'best_val_score': best_val_acc,
            'final_loss': final_loss,
            'final_metrics': final_metrics,
            'train_log': train_log,
            'final_train_metrics': final_train_metrics
        }

        split_results.append(split_result)

        logger.add_split_result(
            split_idx=split_idx,
            best_val_score=best_val_acc,
            final_loss=final_loss,
            final_metrics=final_metrics,
            final_train_metrics=final_train_metrics,
            train_log=train_log,
            test_subject=split.get('test_subject', None),
            test_subjects=split.get('test_subjects', None),
            fold_idx=split.get('fold_idx', None)
        )
        seed_losses.append(final_loss)
        collect_metrics(seed_metrics, final_metrics, final_train_metrics)
    return split_results, seed_losses, seed_metrics


def process_all_splits_traditional_method(splits, model, eeg_data, task_type, is_binary, device, args, logger):
    """Run traditional ML training and evaluation for all splits (CSP_LDA, Xdawn_LDA, PSD_*, etc.)."""
    split_results = []
    seed_losses = []
    seed_metrics = {}
    model_name = args.model_name
    for split_idx, split in enumerate(splits):
        logger.log(f'\n\n===== Processing split {split_idx+1}/{len(splits)} with {model_name} =====')
        if 'test_subject' in split:
            logger.log(f'Test subject: {split["test_subject"]}')
        elif 'fold_idx' in split:
            logger.log(f'Fold index: {split["fold_idx"]}')

        train_data = eeg_data.eeg_data[split['train_mask']]
        test_data = eeg_data.eeg_data[split['test_mask']]
        if task_type == 'regression':
            train_labels = eeg_data.regression_values[split['train_mask']]
            test_labels = eeg_data.regression_values[split['test_mask']]
        else:
            train_labels = eeg_data.labels[split['train_mask']]
            test_labels = eeg_data.labels[split['test_mask']]

        logger.log(f'Train data shape: {train_data.shape}, Test data shape: {test_data.shape}')
        logger.log(f'Starting {model_name} training for split {split_idx+1}')
        model.fit(train_data, train_labels)

        logger.log(f'\nEvaluating {model_name} on test set for split {split_idx+1}:')
        test_predictions = model.predict(test_data)
        if task_type == 'regression':
            test_output_for_metrics = test_predictions
            metrics_to_compute = ['MSE', 'RMSE', 'MAE', 'R2', 'CORR']
            final_metrics = get_metrics(test_output_for_metrics, test_labels, metrics=metrics_to_compute, is_binary=False)
            final_loss = final_metrics.get('MSE', 0.0)
            seed_losses.append(final_loss)
            for key, value in final_metrics.items():
                logger.log(f'Test {key}: {value:.4f}')
            logger.log(f'Loss: {final_loss:.4f}')
            final_train_metrics = {}
            train_predictions = model.predict(train_data)
            train_metrics = get_metrics(train_predictions, train_labels, metrics=['MSE', 'RMSE'], is_binary=False)
            final_train_metrics['MSE'] = train_metrics['MSE']
            final_train_metrics['RMSE'] = train_metrics['RMSE']
            best_val_score = final_metrics.get('RMSE', 0.0)

        else:
            try:
                test_probabilities = model.predict_proba(test_data)
            except (AttributeError, RuntimeError):
                test_probabilities = test_predictions
            metrics_to_compute = ['Acc', 'BAC', "Cohen's Kappa", 'weighted-F1', 'AUROC', 'AUPRC']
            final_metrics = get_metrics(test_probabilities, test_labels, metrics=metrics_to_compute, is_binary=is_binary)
            final_loss = 1.0 - final_metrics.get('accuracy', 0.0)
            seed_losses.append(final_loss)
            if "accuracy" in final_metrics:
                logger.log(f'Test Accuracy: {final_metrics["accuracy"]:.4f}')
            if "balanced_accuracy" in final_metrics:
                logger.log(f'Balanced Accuracy: {final_metrics["balanced_accuracy"]:.4f}')
            if "cohen_kappa" in final_metrics:
                logger.log(f"Cohen's Kappa: {final_metrics['cohen_kappa']:.4f}")
            if "f1_score" in final_metrics:
                logger.log(f'Weighted F1 Score: {final_metrics["f1_score"]:.4f}')
            if "auroc" in final_metrics:
                logger.log(f'AUROC: {final_metrics["auroc"]:.4f}')
            if "auprc" in final_metrics:
                logger.log(f'AUPRC: {final_metrics["auprc"]:.4f}')
            logger.log(f'Loss: {final_loss:.4f}')
            final_train_metrics = {}
            try:
                train_probabilities = model.predict_proba(train_data)
                train_metrics = get_metrics(train_probabilities, train_labels, metrics=['accuracy'], is_binary=is_binary)
                final_train_metrics['accuracy'] = train_metrics['accuracy']
            except (AttributeError, RuntimeError):
                train_predictions = model.predict(train_data)
                train_metrics = get_metrics(train_predictions, train_labels, metrics=['accuracy'], is_binary=is_binary)
                final_train_metrics['accuracy'] = train_metrics['accuracy']
            best_val_score = final_metrics.get('balanced_accuracy', 0.0)

        split_result = {
            'split_idx': split_idx,
            'test_subject': split.get('test_subject', None),
            'test_subjects': split.get('test_subjects', None),
            'fold_idx': split.get('fold_idx', None),
            'best_val_score': best_val_score,
            'final_loss': final_loss,
            'final_metrics': final_metrics,
            'train_log': {'epochs': [0]},
            'final_train_metrics': final_train_metrics
        }
        split_results.append(split_result)
        logger.add_split_result(
            split_idx=split_idx,
            best_val_score=best_val_score,
            final_loss=final_loss,
            final_metrics=final_metrics,
            final_train_metrics=final_train_metrics,
            train_log={'epochs': [0]},
            test_subject=split.get('test_subject', None),
            test_subjects=split.get('test_subjects', None),
            fold_idx=split.get('fold_idx', None)
        )
        collect_metrics(seed_metrics, final_metrics, final_train_metrics)

    return split_results, seed_losses, seed_metrics


# --- Single-seed and multi-seed aggregation ---
def run_single_seed(seed, args, eeg_data, dataset_name, model_name, task_type, is_binary,
                    split_method, task_mode_folder, timestamp, device, logger, preprocessor):
    """Run one seed: split, train, evaluate, save. Returns (seed_result, effective_split_method)."""
    import shutil
    logger.log(f'\n\n===== Running experiment with seed: {seed} =====')
    set_seed(seed)
    logger.log(f'\n===== Starting Fine-tuning with the following parameters =====')
    logger.log(f'Dataset: {dataset_name}, Model: {model_name}, Task type: {task_type}')
    logger.log(f'Split method: {split_method}, Batch size: {args.batch_size}, Epochs: {args.epochs}')
    logger.log(f'Learning rate: {args.lr}, Weight decay: {args.weight_decay}, Seed: {seed}')
    logger.log(f'Fine-tune strategy: {args.finetune_strategy}, Optimizer type: {args.optimizer_type}')
    logger.log(f'Filter bias and BN: {args.filter_bias_and_bn}, Warmup epochs: {args.warmup_epochs}')
    logger.log(f'Min LR: {args.min_lr}')
    logger.log(f'============================================================\n')

    kwargs = build_model_import_kwargs(args)
    model = import_model(model_name, eeg_data, **kwargs)

    if args.task_mode == 'Cross':
        splits = split_dataset_cross(eeg_data, split_method)
    else:
        splits = split_dataset_fewshot(eeg_data, train_percentage=args.train_percentage)
        split_method = f'Fewshot-{int(args.train_percentage*100)}%'
        logger.log(f'Using few-shot dataset split with {int(args.train_percentage*100)}% of each class per subject for training')

    logger.log(f'\n===== Split info =====')
    logger.log(f'Split method: {split_method}')
    logger.log(f'Number of splits: {len(splits)}')
    for i, split in enumerate(splits):
        train_size = np.sum(split['train_mask'])
        test_size = np.sum(split['test_mask'])
        logger.log(f'Split {i+1}: train {train_size} samples, test {test_size} samples')
        if hasattr(eeg_data, 'eeg_data'):
            train_shape = eeg_data.eeg_data[split['train_mask']].shape
            test_shape = eeg_data.eeg_data[split['test_mask']].shape
            logger.log(f'Split {i+1} shapes - train: {train_shape}, test: {test_shape}')

    temp_output_dir = os.path.join('./outputs', 'logs', dataset_name, model_name, task_mode_folder, args.finetune_strategy, timestamp, f'temp_seed_{seed}')
    os.makedirs(temp_output_dir, exist_ok=True)
    logger.experiment_dir = temp_output_dir

    if is_traditional_model(model_name):
        split_results, seed_losses, seed_metrics = process_all_splits_traditional_method(
            splits, model, eeg_data, task_type, is_binary, device, args, logger
        )
    else:
        model = model.to(device)
        split_results, seed_losses, seed_metrics = process_all_splits_dl(
            splits, model, eeg_data, task_type, is_binary, device, args, logger
        )

    seed_avg_loss = np.mean(seed_losses)
    seed_std_loss = np.std(seed_losses)
    logger.log(f'\n\n===== Overall results for seed {seed} =====')
    logger.log('\nTest set metrics:')
    for metric_key in seed_metrics:
        if not metric_key.startswith('train_') and seed_metrics[metric_key]:
            avg_val = np.mean(seed_metrics[metric_key])
            std_val = np.std(seed_metrics[metric_key])
            logger.log(f'Average {metric_key} across all splits: {avg_val:.4f} ± {std_val:.4f}')
    logger.log('\nTraining set metrics:')
    for metric_key in seed_metrics:
        if metric_key.startswith('train_') and seed_metrics[metric_key]:
            avg_val = np.mean(seed_metrics[metric_key])
            std_val = np.std(seed_metrics[metric_key])
            metric_name = metric_key.replace('train_', '')
            logger.log(f'Average train {metric_name}: {avg_val:.4f} ± {std_val:.4f}')

    metric_value = 0.0
    if task_type == 'classification':
        if 'balanced_accuracy' in seed_metrics and seed_metrics['balanced_accuracy']:
            metric_value = np.mean(seed_metrics['balanced_accuracy'])
    elif task_type == 'matching':
        if 'global_top5' in seed_metrics and seed_metrics['global_top5']:
            metric_value = np.mean(seed_metrics['global_top5'])
    else:
        if 'RMSE' in seed_metrics and seed_metrics['RMSE']:
            metric_value = np.mean(seed_metrics['RMSE'])
    metric_str = f"{metric_value:.4f}"
    output_dir = os.path.join('./outputs', 'logs', dataset_name, model_name, task_mode_folder, args.finetune_strategy, timestamp, f'{metric_str}_seed_{seed}')

    if os.path.exists(temp_output_dir):
        shutil.move(temp_output_dir, output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
    logger.experiment_dir = output_dir

    experiment_results = {
        'params': vars(args),
        'seed': seed,
        'dataset': dataset_name,
        'model_name': model_name,
        'task_type': task_type,
        'split_method': split_method,
        'split_results': split_results,
        'avg_loss': seed_avg_loss,
        'std_loss': seed_std_loss
    }
    for metric_key in seed_metrics:
        if not metric_key.startswith('train_') and seed_metrics[metric_key]:
            experiment_results[f'avg_{metric_key.lower()}'] = np.mean(seed_metrics[metric_key])
            experiment_results[f'std_{metric_key.lower()}'] = np.std(seed_metrics[metric_key])
    for metric_key in seed_metrics:
        if metric_key.startswith('train_') and seed_metrics[metric_key]:
            base_name = metric_key.replace('train_', '')
            experiment_results[f'avg_train_{base_name}'] = np.mean(seed_metrics[metric_key])
            experiment_results[f'std_train_{base_name}'] = np.std(seed_metrics[metric_key])
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    logger.log(f'\nSaved experiment results to {results_path}')

    log_path = os.path.join(output_dir, 'training_log.txt')
    logger.log('===== Experiment parameters =====')
    logger.log(f'Dataset: {dataset_name}, Model: {model_name}, Seed: {seed}')
    logger.log(f'Task type: {task_type}, Split method: {split_method}, Total splits: {len(splits)}')
    logger.log('\n===== Preprocessing parameters =====')
    if preprocessor is not None:
        target_fs = getattr(preprocessor, 'target_fs', 'Not set')
        l_freq = getattr(preprocessor, 'l_freq', 'Not set')
        h_freq = getattr(preprocessor, 'h_freq', 'Not set')
        notch_freq = getattr(preprocessor, 'notch_freq', 'Not set')
        norm_method = getattr(preprocessor, 'normalize_method', 'Not set')
        apply_EA = getattr(preprocessor, 'apply_EA', False)
        time_length = getattr(preprocessor, 'time_length', 'Not set')
        param_source = 'from parser' if args.use_preprocessing_params else 'default values'
        logger.log(f'Preprocessing parameters source: {param_source}')
        logger.log(f'Target FS: {target_fs}, Low freq: {l_freq}, High freq: {h_freq}')
        logger.log(f'Notch freq: {notch_freq}, Norm method: {norm_method}')
        logger.log(f'Apply EA: {apply_EA}, Time length: {time_length}')
    else:
        logger.log(f'Target FS: {args.target_fs}, Low freq: {args.l_freq}, High freq: {args.h_freq}')
        logger.log(f'Notch freq: {args.notch_freq}, Norm method: {args.norm_method}')
        logger.log(f'Apply EA: {args.apply_EA}, Time length: {args.time_length}')
    logger.log('\n===== Split results =====')
    for result in split_results:
        if 'test_subject' in result and result['test_subject'] is not None:
            logger.log(f'\nSplit {result["split_idx"]+1} - Subject {result["test_subject"]}:')
        elif 'fold_idx' in result:
            if 'test_subjects' in result and result['test_subjects'] is not None:
                test_subjects_str = ', '.join(map(str, result['test_subjects']))
                logger.log(f'\nSplit {result["split_idx"]+1} - Fold {result["fold_idx"]} (Test Subjects: {test_subjects_str}):')
            else:
                logger.log(f'\nSplit {result["split_idx"]+1} - Fold {result["fold_idx"]}:')
        else:
            logger.log(f'\nSplit {result["split_idx"]+1}:')
        logger.log(f'Best validation score: {result.get("best_val_score", 0.0):.4f}')
        logger.log(f'Final test loss: {result["final_loss"]:.4f}')
        if result.get("final_train_metrics"):
            logger.log('Training metrics:')
            for key, value in result["final_train_metrics"].items():
                logger.log(f'  {key}: {value:.4f}')
        logger.log('Test metrics:')
        for key, value in result["final_metrics"].items():
                logger.log(f'  {key}: {value:.4f}')

    logger.log('\n===== Overall results =====')
    logger.log(f'Average loss: {seed_avg_loss:.4f} ± {seed_std_loss:.4f}')
    logger.log('\nAverage test metrics for this seed:')
    for metric_key in seed_metrics:
        if not metric_key.startswith('train_') and seed_metrics[metric_key]:
            avg_val = np.mean(seed_metrics[metric_key])
            std_val = np.std(seed_metrics[metric_key])
            logger.log(f'Average {metric_key}: {avg_val:.4f} ± {std_val:.4f}')
    logger.save_log(log_path)
    logger.log(f'Saved training log to {log_path}')

    results_txt_path = os.path.join(output_dir, 'results.txt')
    experiment_info = {
        'params': vars(args),
        'seed': seed,
        'dataset': dataset_name,
        'model_name': model_name,
        'task_type': task_type,
        'split_method': split_method,
        'timestamp': timestamp,
        'device': str(device)
    }
    logger.save_results_as_text(results_txt_path, split_results=split_results, experiment_info=experiment_info)
    logger.log(f'Saved results text file to {results_txt_path}')

    seed_result = {
        'seed': seed,
        'split_results': split_results,
        'avg_loss': seed_avg_loss,
        'std_loss': seed_std_loss
    }
    return seed_result, split_method


def _aggregate_and_save_all_seeds(args, all_seed_results, dataset_name, model_name, task_type,
                                  split_method, task_mode_folder, timestamp, device, logger):
    """Aggregate results across seeds and save CSV after all seeds are done."""
    logger.log(f'\n\n===== All seeds completed =====')
    overall_avg_loss = np.mean([sr['avg_loss'] for sr in all_seed_results])
    overall_std_loss = np.std([sr['avg_loss'] for sr in all_seed_results])
    logger.log(f'\n\n===== Overall results across all seeds =====')
    logger.log(f'Average loss across all seeds: {overall_avg_loss:.4f} ± {overall_std_loss:.4f}')

    all_metrics = {}
    for seed_result in all_seed_results:
        for split in seed_result['split_results']:
            for metric_name, metric_value in split['final_metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
    if all_metrics:
        logger.log('\nAverage test metrics across all seeds:')
        for metric_name, values in all_metrics.items():
            avg_value = np.mean(values)
            std_value = np.std(values)
            logger.log(f'Average {metric_name}: {avg_value:.4f} ± {std_value:.4f}')

    timestamp_dir = os.path.join('./outputs', 'logs', dataset_name, model_name, task_mode_folder, args.finetune_strategy, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    experiment_info = {
        'params': vars(args),
        'dataset': dataset_name,
        'model_name': model_name,
        'task_type': task_type,
        'split_method': split_method,
        'timestamp': timestamp,
        'device': str(device),
        'num_seeds': len(args.seeds),
        'seeds': args.seeds,
        'overall_avg_loss': overall_avg_loss,
        'overall_std_loss': overall_std_loss
    }

    all_seed_results_list = []
    for s in args.seeds:
        seed_result = None
        for result in all_seed_results:
            if result.get('seed') == s:
                seed_result = result
                break
        if seed_result:
            all_seed_results_list.append({
                'seed': s,
                'split_results': seed_result.get('split_results', []),
                'avg_loss': seed_result.get('avg_loss', 0),
                'std_loss': seed_result.get('std_loss', 0)
            })

    avg_accuracy = 0.0
    accuracy_count = 0
    if all_seed_results_list:
        for seed_result in all_seed_results_list:
            for split in seed_result.get('split_results', []):
                metrics = split.get('final_metrics', {})
                if task_type == 'matching':
                    accuracy = metrics.get('top5', metrics.get('accuracy', 0))
                else:
                    accuracy = metrics.get('balanced_accuracy', metrics.get('accuracy', 0))
                if isinstance(accuracy, (int, float)):
                    avg_accuracy += accuracy
                    accuracy_count += 1
        if accuracy_count > 0:
            avg_accuracy = avg_accuracy / accuracy_count
    avg_accuracy_percent = f"{avg_accuracy * 100:.2f}"
    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"all_seeds_results_{avg_accuracy_percent}_{current_timestamp}.csv"
    results_csv_path = os.path.join(timestamp_dir, csv_filename)

    logger.save_results_as_csv(
        file_path=results_csv_path,
        all_seed_results=all_seed_results_list,
        experiment_info=experiment_info
    )
    logger.log(f'Saved combined CSV results for all seeds to {results_csv_path}')


# Main
def main():
    """Parse args, load data and preprocess, run all seeds (train/eval/save), then aggregate and manage logs."""
    parser = create_parser()
    
    # Parse CLI
    args = parser.parse_args()
    
    # Timestamp for unique experiment dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Logger without output dir yet; dir is set later from dataset/model
    logger = Logger(output_dir=None, experiment_name=None, verbose=True)
    set_global_logger(logger)
    
    # Device selection (use specified GPU ID)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpuid}')
        torch.cuda.set_device(args.gpuid)
        device_info = f'cuda:{args.gpuid} ({torch.cuda.get_device_name(args.gpuid)})'
    else:
        device = torch.device('cpu')
        device_info = 'cpu'
    logger.log(f'Using device: {device_info}')
    
    # Single-dataset run
    dataset_name = args.dataset
    logger.log(f'\n\n==================================================================')
    logger.log(f'===== Starting experiments for dataset: {dataset_name} =====')
    logger.log(f'==================================================================')
        
    # Load dataset (no model-specific preprocessing yet)
    load_jobs = args.data_workers if args.data_workers > 1 else None
    dataset_result = import_dataset(dataset_name, n_jobs=load_jobs)

    if dataset_result is None:
        logger.log(f'Failed to import dataset: {dataset_name}')
    
    # Single-model run
    model_name = args.model_name
    logger.log(f'\n\n==================================================================')
    logger.log(f'===== Starting experiments with model: {model_name} on dataset: {dataset_name} =====')
    logger.log(f'==================================================================')

    # Dataset info
    eeg_data_original, task_type, is_binary, split_method = dataset_result
    
    logger.log(f'\n===== Raw dataset info =====')
    log_dataset_info(logger, eeg_data_original, dataset_name, task_type, is_binary, split_method)
    
        
    eeg_data = copy.deepcopy(eeg_data_original)

    # Apply model-specific preprocessing
    if args.use_preprocessing_params:
        preprocessing_kwargs = {
            'target_fs': args.target_fs,
            'l_freq': args.l_freq,
            'h_freq': args.h_freq,
            'notch_freq': args.notch_freq,
            'normalize_method': args.norm_method,
            'apply_EA': args.apply_EA,
            'time_length': args.time_length,
            'n_workers': args.data_workers
        }
        logger.log(f'Using preprocessing parameters from parser: {preprocessing_kwargs}')
        preprocessor = import_preprocessor(model_name, **preprocessing_kwargs)
        eeg_data = apply_preprocessor(eeg_data, preprocessor, model_name, dataset_name, task_mode=args.task_mode, train_percentage=args.train_percentage)
    else:
        preprocessor = import_preprocessor(model_name, n_workers=args.data_workers)
        eeg_data = apply_preprocessor(eeg_data, preprocessor, model_name, dataset_name, task_mode=args.task_mode, train_percentage=args.train_percentage)
    
    logger.log(f'\n===== Preprocessed dataset info =====')
    log_dataset_info(logger, eeg_data, dataset_name, task_type, is_binary, split_method)

    task_mode_folder = f"Fewshot-{int(args.train_percentage * 100)}%" if args.task_mode == 'Fewshot' else args.task_mode

    # Multi-seed loop
    all_seed_results = []
    for seed in args.seeds:
        seed_result, split_method = run_single_seed(
            seed, args, eeg_data, dataset_name, model_name, task_type, is_binary,
            split_method, task_mode_folder, timestamp, device, logger, preprocessor
        )
        all_seed_results.append(seed_result)

    if all_seed_results:
        _aggregate_and_save_all_seeds(
            args, all_seed_results, dataset_name, model_name, task_type,
            split_method, task_mode_folder, timestamp, device, logger
        )

    # Prune old timepoint dirs (see logger.manage_timepoint_directories)
    logger.manage_timepoint_directories(dataset_name, model_name, task_mode_folder)


if __name__ == '__main__':
    main()