import torch
import numpy as np
import time
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import get_metrics
from utils.logger import get_logger
from utils.optimizer import cosine_scheduler, get_layer_decay_values, create_optimizer


class ClipLoss(nn.Module):
    def __init__(self, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss



def extract_prediction(outputs):
    """Extract prediction tensor from model output (tuple or single tensor)."""
    return outputs[1] if isinstance(outputs, tuple) else outputs


def compute_and_log_metrics(predictions, labels, task_type, is_binary, logger, phase='Train', k=None):
    """Compute metrics (Acc, BAC, MSE, etc.) and log them; matching task metrics are computed in evaluate_model."""
    labels_np = np.array(labels)
    if logger is None:
        logger = get_logger()
    
    if task_type == 'matching':
        # Matching task metrics are computed in evaluate_model and duplicate computation is skipped
        metrics = {}
    elif task_type == 'regression':
        metrics = get_metrics(
            np.array(predictions),
            labels_np,
            metrics=['MSE', 'RMSE', 'MAE', 'R2', 'CORR'],
            is_binary=False
        )
    else:
        output_for_metrics = np.array(predictions)
        metrics = get_metrics(
            output_for_metrics,
            labels_np,
            metrics=['Acc', 'BAC', "Cohen's Kappa", 'weighted-F1'] + (['AUROC', 'AUPRC'] if is_binary else []),
            is_binary=is_binary
        )
    for key, value in metrics.items():
        logger.log(f'{phase} {key}: {value:.4f}')
    
    return metrics

def train_model(model, train_loader, test_loader, task_type, device, args, logger=None, k_values=None, topn_values=None):
    """Train model and return (model, best_val_metric, train_log).
    Note: best_val_metric is the best validation metric so far (accuracy for classification/matching, loss for regression)."""
    epochs = args.epochs
    if logger is None:
        logger = get_logger()
    training_log_path = None
    last_save_time = time.time()
    log_save_interval = 300
    last_log_index = 0
    if hasattr(logger, 'experiment_dir') and logger.experiment_dir is not None:
        training_log_path = os.path.join(logger.experiment_dir, 'training_log.txt')
        logger.save_log(training_log_path)
    if task_type == 'regression':
        is_binary = False
        criterion = torch.nn.MSELoss()
    elif task_type == 'matching':
        is_binary = False
        criterion = ClipLoss()
        if not hasattr(model, 'loss_scale'):
            model.loss_scale = torch.tensor(100.0, requires_grad=True, device=device)
        if not hasattr(model, 'logit_scale'):
            model.logit_scale = model.loss_scale
    else:
        num_classes = len(np.unique(train_loader.dataset.tensors[1].cpu().numpy()))
        is_binary = (num_classes == 2)
        label_smoothing = getattr(args, 'label_smoothing', 0.0)
        weight = None
        if hasattr(args, 'class_weights') and args.class_weights:
            train_labels = train_loader.dataset.tensors[1].cpu().numpy()
            class_counts = np.bincount(train_labels)
            class_weights = len(train_labels) / (len(class_counts) * class_counts)
            weight = torch.tensor(class_weights, dtype=torch.float).to(device)
            logger.log(f'Calculated class weights: {class_weights}')
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight)
    num_training_steps_per_epoch = len(train_loader)
    lr_schedule_values = None
    wd_schedule_values = None
    if args.use_lr_scheduler:
        actual_warmup_epochs = min(args.warmup_epochs, args.epochs)
        
        lr_schedule_values = cosine_scheduler(
            args.lr, args.min_lr , args.epochs, num_training_steps_per_epoch,
            warmup_epochs=actual_warmup_epochs,
            warmup_steps=-1,
        )
        if args.use_weight_decay_schedule:
            weight_decay_end = args.min_weight_decay
            wd_schedule_values = cosine_scheduler(
                args.weight_decay, weight_decay_end, args.epochs, num_training_steps_per_epoch
            )
    get_num_layer, get_layer_scale = get_layer_decay_values(model, args.model_name, args.layer_decay)
    optimizer = create_optimizer(
        model=model,
        opt=args.optimizer_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt_eps=args.opt_eps,
        momentum=args.momentum,
        get_num_layer=get_num_layer,
        get_layer_scale=get_layer_scale,
        filter_bias_and_bn=args.filter_bias_and_bn
    )
    best_val_acc = 0.0 if task_type in ['classification', 'matching'] else float('inf')
    best_model_wts = None
    train_log = {
        'epochs': [],
        'loss': [],
        'metrics': []
    }
    global_step = 0
    for epoch in range(epochs):
        logger.log(f'\nEpoch {epoch+1}/{epochs}')
        logger.log('-' * 10)
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for batch_data in progress_bar:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if task_type == 'matching':
                    eeg_data, labels, candidate_features = batch_data
                    eeg_data = eeg_data.to(device)
                    labels = labels.to(device)
                    candidate_features = candidate_features.to(device)
                    eeg_features = model(eeg_data)
                    logit_scale = model.loss_scale
                    loss = criterion(eeg_features, candidate_features, logit_scale)
                    running_loss += loss.item() * eeg_data.size(0)
                    all_labels.extend(labels.data.cpu().numpy())
                    logits = logit_scale * eeg_features @ candidate_features.T
                    predicted = torch.argmax(logits, dim=1)
                    all_preds.extend(predicted.data.cpu().numpy())
                    all_probs.extend(logits.data.cpu().numpy())
                    
                    inputs_size = eeg_data.size(0)
                else:
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    pred = extract_prediction(outputs)
                    if task_type == 'regression':
                        if pred.ndim == 2 and pred.shape[1] == 1:
                            pred = pred.squeeze(1)
                        loss = criterion(pred, labels.float())
                    else:
                        loss = criterion(pred, labels)
                    running_loss += loss.item() * inputs.size(0)
                    all_labels.extend(labels.data.cpu().numpy())
                    pred_output = extract_prediction(outputs)
                    if task_type == 'regression':
                        preds = pred_output.cpu().detach().numpy()
                        if preds.ndim == 0:
                            preds = np.array([preds])
                        elif preds.ndim == 2 and preds.shape[1] == 1:
                            preds = preds.squeeze(1)
                        probs = preds  
                    else:
                        probs = torch.softmax(pred_output, dim=1).cpu().detach().numpy()
                        preds = np.argmax(probs, axis=1)
                    
                    all_preds.extend(preds)
                    all_probs.extend(probs)
                    
                    inputs_size = inputs.size(0)
                loss.backward()
                grad_norm = None
                if hasattr(args, 'use_grad_clipping') and args.use_grad_clipping and args.clip_grad_norm is not None:
                    parameters_with_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                    if parameters_with_grad:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            parameters_with_grad,
                            max_norm=args.clip_grad_norm,
                            norm_type=2.0
                        ).item()
                optimizer.step()
                global_step += 1
            current_time = time.time()
            if training_log_path is not None and (current_time - last_save_time >= log_save_interval):
                logger.save_log_incremental(training_log_path, last_log_index)
                last_log_index = len(logger.log_history)
                last_save_time = current_time
            postfix = {'loss': loss.item()}
            if grad_norm is not None:
                postfix['grad_norm'] = grad_norm
            if wd_schedule_values is not None:
                wd_value = next((pg['weight_decay'] for pg in optimizer.param_groups if pg['weight_decay'] > 0), 0)
                postfix['wd'] = wd_value
            progress_bar.set_postfix(postfix)
        epoch_loss = running_loss / len(train_loader.dataset)
        logger.log(f'Train Loss: {epoch_loss:.4f}')
        train_metrics = {}
        train_metrics['loss'] = epoch_loss
        if task_type != 'matching':
            train_metrics.update(compute_and_log_metrics(all_probs, all_labels, task_type, is_binary, logger, phase='Train'))
        
        if task_type == 'matching':
            all_img_features = []
            all_eeg_features = []
            all_true_labels = []
            
            model.eval()
            with torch.no_grad():
                for batch_data in tqdm(train_loader, desc='Collecting all training features', leave=False):
                    if task_type == 'matching':
                        eeg_data, labels, img_features = batch_data
                        eeg_data = eeg_data.to(device)
                        img_features = img_features.to(device)
                        eeg_features_batch = model(eeg_data)
                    
                        all_img_features.append(img_features)
                        all_eeg_features.append(eeg_features_batch)
                        all_true_labels.extend(labels.cpu().numpy())
            
            all_img_features = torch.cat(all_img_features, dim=0)
            all_eeg_features = torch.cat(all_eeg_features, dim=0)
            all_true_labels = np.array(all_true_labels)
            all_unique_labels = np.unique(all_true_labels)
            all_labels_set = set(all_unique_labels)
            all_eeg_labels = all_true_labels.copy()
            if hasattr(model, 'logit_scale'):
                logit_scale = model.logit_scale
            elif hasattr(model, 'loss_scale'):
                logit_scale = model.loss_scale
            else:
                logit_scale = 1.0
            global_topn_correct = {}
            global_topn_values = topn_values
            for n in global_topn_values:
                global_topn_correct[n] = 0
            global_logits = logit_scale * all_eeg_features @ all_img_features.T
            for idx, true_label in enumerate(all_eeg_labels):
                current_logits = global_logits[idx]
                for n in global_topn_values:
                    topn_indices = torch.topk(current_logits, n, largest=True)[1]
                    if true_label in all_true_labels[topn_indices.tolist()]:
                        global_topn_correct[n] += 1
            total_samples = len(all_eeg_labels)
            for n in global_topn_values:
                global_topn_acc = global_topn_correct[n] / total_samples
                train_metrics[f'global_top{n}'] = global_topn_acc
                logger.log(f'Train global_top{n}: {global_topn_acc:.4f}')
            for k in k_values:
                correct = 0
                total = 0
                if k > len(all_true_labels):
                    logger.log(f'Warning: k={k} exceeds training sample count {len(all_true_labels)}, adjusting to sample count')
                    k = len(all_true_labels)
                valid_topn_values = [1]
                topn_correct = {1: 0}
                
        # Save training log
        train_log['epochs'].append(epoch+1)
        train_log['loss'].append(epoch_loss)
        train_log['metrics'].append(train_metrics)
        val_metrics = evaluate_model(model, test_loader, criterion, task_type, is_binary, device, logger, k_values, topn_values)
        val_loss = val_metrics.get('loss', 0.0)
        update_optimization_params(optimizer, global_step, lr_schedule_values, wd_schedule_values, epoch, logger)
        is_best_model = check_best_model(val_loss, val_metrics, best_val_acc, task_type, logger)
        if is_best_model:
            if task_type == 'regression':
                best_val_acc = val_loss
            elif task_type == 'matching':
                best_val_acc = val_metrics.get('top5', 0.0)
            else:
                best_val_acc = val_metrics.get('accuracy', 0.0)

    
    logger.log('\nTraining completed')
    logger.log(f'Best validation {"loss" if task_type == "regression" else "accuracy"}: {best_val_acc:.4f}')
    if training_log_path is not None:
        logger.save_log_incremental(training_log_path, last_log_index)
        last_log_index = len(logger.log_history)
        last_save_time = time.time()
    
    return model, best_val_acc, train_log


def update_optimization_params(optimizer, global_step, lr_schedule_values, wd_schedule_values, epoch, logger):
    """Apply LR and weight-decay schedules to optimizer param_groups."""
    if lr_schedule_values is None and wd_schedule_values is None:
        return
    if lr_schedule_values is not None:
        step_idx = min(global_step, len(lr_schedule_values) - 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule_values[step_idx] * param_group.get("lr_scale", 1.0)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log(f'Epoch {epoch+1} - Learning Rate: {current_lr:.8f}')
    if wd_schedule_values is not None:
        step_idx = min(global_step, len(wd_schedule_values) - 1)
        for param_group in optimizer.param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[step_idx]


def check_best_model(val_loss, val_metrics, best_val_acc, task_type, logger):
    """Return True if current validation is the best so far (accuracy for classification/matching, loss for regression)."""
    if task_type == 'regression':
        if val_loss < best_val_acc:
            logger.log(f'New best validation loss: {val_loss:.4f}')
            return True
    elif task_type == 'matching':
        val_acc = val_metrics.get('top5', 0.0)
        if val_acc > best_val_acc:
            logger.log(f'New best validation top5: {val_acc:.4f}')
            return True
    else:
        val_acc = val_metrics.get('accuracy', 0.0)
        if val_acc > best_val_acc:
            logger.log(f'New best validation accuracy: {val_acc:.4f}')
            return True
    return False


def evaluate_model(model, dataloader, criterion, task_type, is_binary=False, device=None, logger=None, k_values=None, topn_values=None):
    """Evaluate model on dataloader; for matching task also computes global top-n and v{k} candidate-set metrics."""
    if logger is None:
        logger = get_logger()
    
    model.eval()
    # Get logit_scale once
    if hasattr(model, 'logit_scale'):
        logit_scale = model.logit_scale
    elif hasattr(model, 'loss_scale'):
        logit_scale = model.loss_scale
    else:
        logit_scale = 1.0
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    training_log_path = None
    last_save_time = time.time()
    log_save_interval = 300
    last_log_index = 0
    if hasattr(logger, 'experiment_dir') and logger.experiment_dir is not None:
        training_log_path = os.path.join(logger.experiment_dir, 'training_log.txt')
    
    progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for batch_data in progress_bar:
            if task_type == 'matching':
                eeg_data, labels, candidate_features = batch_data
                if device is not None:
                    eeg_data = eeg_data.to(device)
                    labels = labels.to(device)
                    candidate_features = candidate_features.to(device)
                eeg_features = model(eeg_data)
                loss = criterion(eeg_features, candidate_features, logit_scale)
                running_loss += loss.item() * eeg_data.size(0)
                all_labels.extend(labels.data.cpu().numpy())
                logits = logit_scale * eeg_features @ candidate_features.T
                predicted = torch.argmax(logits, dim=1)
                all_preds.extend(predicted.data.cpu().numpy())
                all_probs.extend(logits.data.cpu().numpy())
            else:
                inputs, labels = batch_data
                if device is not None:
                    inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                pred = extract_prediction(outputs)
                if task_type == 'regression':
                    if pred.ndim == 2 and pred.shape[1] == 1:
                        pred = pred.squeeze(1)
                    loss = criterion(pred, labels.float())
                else:
                    loss = criterion(pred, labels)
                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.data.cpu().numpy())
                pred_output = extract_prediction(outputs)
                if task_type == 'regression':
                    preds = pred_output.cpu().detach().numpy()
                    if preds.ndim == 0:
                        preds = np.array([preds])
                    elif preds.ndim == 2 and preds.shape[1] == 1:
                        preds = preds.squeeze(1)
                    probs = preds
                else:
                    probs = torch.softmax(pred_output, dim=1).cpu().detach().numpy()
                    preds = np.argmax(probs, axis=1)
                
                all_preds.extend(preds)
                all_probs.extend(probs)
            current_time = time.time()
            if training_log_path is not None and (current_time - last_save_time >= log_save_interval):
                logger.save_log_incremental(training_log_path, last_log_index)
                last_log_index = len(logger.log_history)
                last_save_time = current_time
    avg_loss = running_loss / len(dataloader.dataset)
    logger.log(f'Eval Loss: {avg_loss:.4f}')
    metrics = compute_and_log_metrics(all_probs, all_labels, task_type, is_binary, logger, phase='Eval')
    if task_type == 'matching':
        all_img_features = []
        all_eeg_features = []
        all_true_labels = []
        
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc='Collecting all features', leave=False):
                if task_type == 'matching':
                    eeg_data, labels, img_features = batch_data
                    if device is not None:
                        eeg_data = eeg_data.to(device)
                        img_features = img_features.to(device)
                    eeg_features_batch = model(eeg_data)
                    
                    all_img_features.append(img_features)
                    all_eeg_features.append(eeg_features_batch)
                    all_true_labels.extend(labels.cpu().numpy())
        
        all_img_features = torch.cat(all_img_features, dim=0)
        all_eeg_features = torch.cat(all_eeg_features, dim=0)
        all_true_labels = np.array(all_true_labels)
        all_unique_labels = np.unique(all_true_labels)
        all_labels_set = set(all_unique_labels)
        all_eeg_labels = all_true_labels.copy()
        current_sample_idx = 0
        global_topn_correct = {}
        global_topn_values = topn_values
        for n in global_topn_values:
            global_topn_correct[n] = 0
        global_logits = logit_scale * all_eeg_features @ all_img_features.T
        for idx, true_label in enumerate(all_eeg_labels):
            current_logits = global_logits[idx]
            for n in global_topn_values:
                topn_indices = torch.topk(current_logits, n, largest=True)[1]
                if true_label in all_true_labels[topn_indices.tolist()]:
                    global_topn_correct[n] += 1
        total_samples = len(all_eeg_labels)
        for n in global_topn_values:
                global_topn_acc = global_topn_correct[n] / total_samples
                metrics[f'global_top{n}'] = global_topn_acc
                logger.log(f'Eval global_top{n}: {global_topn_acc:.4f}')
        for k in k_values:
            correct = 0
            total = 0
            if k > len(all_true_labels):
                logger.log(f'Warning: k={k} exceeds sample count {len(all_true_labels)}, adjusting to sample count')
                k = len(all_true_labels)
            valid_topn_values = [1]
            topn_correct = {1: 0}
            model.eval()
            with torch.no_grad():
                for current_sample_idx in tqdm(range(len(all_eeg_features)), desc=f'Evaluating v{k}', leave=False):
                    current_eeg_feature = all_eeg_features[current_sample_idx]
                    current_true_label = all_true_labels[current_sample_idx]
                    different_label_indices = [i for i in range(len(all_true_labels)) if i != current_sample_idx and all_true_labels[i] != current_true_label]
                    if len(different_label_indices) < k-1:
                        selected_neg_indices = random.choices(different_label_indices, k=k-1)
                    else:
                        selected_neg_indices = random.sample(different_label_indices, k=k-1)
                    selected_features = []
                    selected_labels = []
                    for neg_idx in selected_neg_indices:
                        selected_features.append(all_img_features[neg_idx])
                        selected_labels.append(all_true_labels[neg_idx])
                    selected_features.append(all_img_features[current_sample_idx])
                    selected_labels.append(current_true_label)
                    selected_img_features = torch.stack(selected_features, dim=0)
                    logits_img = logit_scale * current_eeg_feature @ selected_img_features.T
                    predicted_idx = torch.argmax(logits_img).item()
                    predicted_label = selected_labels[predicted_idx]
                    if predicted_label == current_true_label:
                        correct += 1
                    for n in valid_topn_values:
                        topn_indices = torch.topk(logits_img, n, largest=True)[1]
                        if current_true_label in [selected_labels[i] for i in topn_indices.tolist()]:
                            topn_correct[n] += 1
                    total += 1
            if total > 0:
                for n in valid_topn_values:
                    vk_topn_acc = topn_correct[n] / total
                    metrics[f'v{k}_top{n}'] = vk_topn_acc
                    logger.log(f'Eval v{k}_top{n}: {vk_topn_acc:.4f}')
            else:
                for n in valid_topn_values:
                    metrics[f'v{k}_top{n}'] = 0.0
                    logger.log(f'Eval v{k}_top{n}: 0.0000 (no samples evaluated)')
    metrics['loss'] = avg_loss
    if training_log_path is not None:
        logger.save_log_incremental(training_log_path, last_log_index)
    
    return metrics