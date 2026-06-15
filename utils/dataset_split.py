"""Dataset splitting (LOSO, k-fold, few-shot) and DataLoader creation."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold


def split_dataset_cross(eeg_data, split_method, shuffle=False):
    """Split dataset by LOSO (leave-one-subject-out) or k-fold; if eeg_data.usage exists, train/test masks respect it."""
    splits = []

    has_usage = hasattr(eeg_data, 'usage') and eeg_data.usage is not None

    if split_method == 'LOSO' or split_method.startswith('LOSO'):
        unique_subjects = np.unique(eeg_data.subject_ids)
        unique_subjects = np.array(sorted(unique_subjects, key=lambda x: float(x)))

        for test_subject in unique_subjects:
            if has_usage:
                train_mask = (eeg_data.subject_ids != test_subject) & (eeg_data.usage == 'train')
                test_mask = (eeg_data.subject_ids == test_subject) & (eeg_data.usage == 'test')
            else:
                train_mask = eeg_data.subject_ids != test_subject
                test_mask = eeg_data.subject_ids == test_subject
            splits.append({
                'train_mask': train_mask,
                'test_mask': test_mask,
                'test_subject': test_subject
            })
    elif split_method.endswith('-fold'):
        n_folds = int(split_method.split('-')[0])
        unique_subjects = np.unique(eeg_data.subject_ids)
        unique_subjects = np.array(sorted(unique_subjects, key=lambda x: float(x)))
        kf = KFold(n_splits=n_folds, shuffle=True)

        for fold_idx, (train_subject_indices, test_subject_indices) in enumerate(kf.split(unique_subjects)):
            train_subjects = unique_subjects[train_subject_indices]
            test_subjects = unique_subjects[test_subject_indices]

            if has_usage:
                train_mask = np.isin(eeg_data.subject_ids, train_subjects) & (eeg_data.usage == 'train')
                test_mask = np.isin(eeg_data.subject_ids, test_subjects) & (eeg_data.usage == 'test')
            else:
                train_mask = np.isin(eeg_data.subject_ids, train_subjects)
                test_mask = np.isin(eeg_data.subject_ids, test_subjects)

            splits.append({
                'train_mask': train_mask,
                'test_mask': test_mask,
                'fold_idx': fold_idx,
                'train_subjects': train_subjects,
                'test_subjects': test_subjects
            })

            print(f'Fold {fold_idx}: Train subjects {len(train_subjects)}, Test subjects {len(test_subjects)}')

    return splits


def split_dataset_fewshot(eeg_data, train_percentage=0.3):
    """Few-shot splits per subject; uses eeg_data.usage when present, else splits by train_percentage per subject/label."""
    splits = []
    unique_subjects = np.unique(eeg_data.subject_ids)
    has_regression = hasattr(eeg_data, 'regression_values') and eeg_data.regression_values is not None
    has_labels = hasattr(eeg_data, 'labels') and eeg_data.labels is not None
    has_usage = hasattr(eeg_data, 'usage') and eeg_data.usage is not None

    for subject_idx, subject in enumerate(unique_subjects):
        train_mask = np.zeros_like(eeg_data.subject_ids, dtype=bool)
        test_mask = np.zeros_like(eeg_data.subject_ids, dtype=bool)

        subject_mask = eeg_data.subject_ids == subject
        subject_indices = np.where(subject_mask)[0]

        if has_usage:
            train_mask = subject_mask & (eeg_data.usage == 'train')
            test_mask = subject_mask & (eeg_data.usage == 'test')
        else:
            if has_regression and not has_labels:
                total_samples = len(subject_indices)
                train_size = max(1, min(int(total_samples * train_percentage), total_samples - 1))
                subject_indices = np.random.permutation(subject_indices)
                train_indices = subject_indices[:train_size]
                test_indices = subject_indices[train_size:]
                train_mask[train_indices] = True
                test_mask[test_indices] = True

            elif has_labels:
                subject_labels = eeg_data.labels[subject_mask]
                for label in np.unique(subject_labels):
                    label_mask = subject_labels == label
                    label_indices = subject_indices[label_mask]
                    train_size = max(1, int(len(label_indices) * train_percentage))
                    train_indices = label_indices[:train_size]
                    test_indices = label_indices[train_size:]
                    train_mask[train_indices] = True
                    test_mask[test_indices] = True

        split_method = f'Fewshot-{train_percentage*100:.0f}%'
        if has_usage:
            split_method = f'Fewshot-by-usage'

        splits.append({
            'train_mask': train_mask,
            'test_mask': test_mask,
            'split_method': split_method,
            'subject_id': subject,
            'fold_idx': subject_idx
        })

    return splits


def create_dataloaders(eeg_data, train_mask, test_mask, batch_size, device, num_workers=4, pin_memory=None):
    """Build train and test DataLoaders for classification, regression, or matching (with img_feature)."""
    if hasattr(eeg_data, 'eeg_data'):
        data = eeg_data.eeg_data
        task_type = eeg_data.dataset_type

        if task_type == 'regression':
            if hasattr(eeg_data, 'regression_values') and eeg_data.regression_values is not None:
                labels = eeg_data.regression_values
            else:
                raise ValueError('Regression task requires regression_values in EEGData object')
        elif task_type == 'classification' or task_type == 'matching':
            if hasattr(eeg_data, 'labels') and eeg_data.labels is not None:
                labels = eeg_data.labels
            else:
                raise ValueError('Classification or matching task requires labels in EEGData object')

    train_data, train_labels = data[train_mask], labels[train_mask]
    test_data, test_labels = data[test_mask], labels[test_mask]

    if train_labels.ndim > 1 and train_labels.shape[1] == 1:
        train_labels, test_labels = train_labels.flatten(), test_labels.flatten()

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

    if train_data_tensor.dim() == 2:
        train_data_tensor = train_data_tensor.unsqueeze(0)
        test_data_tensor = test_data_tensor.unsqueeze(0)

    if task_type == 'regression':
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
    elif task_type == 'classification' or task_type == 'matching':
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    if task_type == 'matching':
        if hasattr(eeg_data, 'img_feature') and eeg_data.img_feature is not None:
            img_features = eeg_data.img_feature
            if isinstance(img_features, np.ndarray):
                img_features_tensor = torch.tensor(img_features, dtype=torch.float32)
            else:
                img_features_tensor = img_features

            if img_features_tensor.dim() == 1:
                img_features_tensor = img_features_tensor.unsqueeze(0)

            train_img_features = img_features_tensor[train_mask]
            test_img_features = img_features_tensor[test_mask]

            if num_workers == 0:
                train_img_features = train_img_features.to(device)
                test_img_features = test_img_features.to(device)

            train_dataset = TensorDataset(train_data_tensor, train_labels_tensor, train_img_features)
            test_dataset = TensorDataset(test_data_tensor, test_labels_tensor, test_img_features)
        else:
            raise ValueError('Matching task requires img_feature in EEGData object')
    else:
        if num_workers == 0:
            train_data_tensor = train_data_tensor.to(device)
            test_data_tensor = test_data_tensor.to(device)
            train_labels_tensor = train_labels_tensor.to(device)
            test_labels_tensor = test_labels_tensor.to(device)

        train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
        test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader
