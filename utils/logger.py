import os
import sys
import json
import shutil
import numpy as np
from datetime import datetime
import pandas as pd

class Logger:
    """Logger for training/eval: log history, split results, save to text/CSV."""
    def __init__(self, output_dir=None, experiment_name=None, verbose=True):
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.verbose = verbose
        self.train_log = {'epochs': [], 'loss': [], 'metrics': []}
        self.split_results = []
        self.log_history = []
        if self.output_dir:
            self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
            os.makedirs(self.experiment_dir, exist_ok=True)
        else:
            self.experiment_dir = None

    def log(self, message):
        """Print to console (if verbose) and append to log_history."""
        if self.verbose:
            print(message)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_history.append(f"[{timestamp}] {message}")
    
    def add_split_result(self, split_idx, best_val_score=None, final_loss=None,
                         final_metrics=None, final_train_metrics=None, train_log=None,
                         test_subject=None, fold_idx=None, **kwargs):
        """Append one split result (classification or regression)."""
        split_result = {
            'split_idx': split_idx,
            'best_val_score': best_val_score,
            'final_loss': final_loss,
            'final_metrics': final_metrics or {},
            'final_train_metrics': final_train_metrics or {},
            'train_log': train_log or self.train_log
        }
        if test_subject is not None:
            split_result['test_subject'] = test_subject
        if fold_idx is not None:
            split_result['fold_idx'] = fold_idx
        for key, value in kwargs.items():
            split_result[key] = value
        self.split_results.append(split_result)
    
    def save_log(self, file_path, append=False):
        """Write full log_history to file; append=True appends, else overwrites."""
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as f:
            for log_entry in self.log_history:
                f.write(f"{log_entry}\n")
        
        if not append:
            self.log(f'Log saved to {file_path}')
    
    def save_log_incremental(self, file_path, start_index=0):
        """Append log_history[start_index:] to file."""
        if start_index >= len(self.log_history):
            return
            
        with open(file_path, 'a', encoding='utf-8') as f:
            for log_entry in self.log_history[start_index:]:
                f.write(f"{log_entry}\n")
    
    def save_results_as_text(self, file_path=None, filename='results.txt', split_results=None, experiment_info=None):
        """Save experiment summary and per-split metrics to a text file."""
        if not file_path:
            if not self.experiment_dir:
                self.log("Warning: No output directory specified, results text not saved.")
                return
            file_path = os.path.join(self.experiment_dir, filename)
        results_to_use = split_results if split_results is not None else self.split_results
        if not results_to_use:
            self.log("Warning: No split results available to save as text.")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("===== Experiment Summary =====\n")
                f.write("No split results available.\n")
            return
        avg_metrics = self._calculate_avg_metrics(results_to_use)
        task_type = 'regression'
        for result in results_to_use:
            if 'final_metrics' in result and isinstance(result['final_metrics'], dict):
                if 'accuracy' in result['final_metrics']:
                    task_type = 'classification'
                    break
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if experiment_info and 'params' in experiment_info:
                f.write("\nComplete Parameters for Copy-Paste:\n")
                cmd_params = []
                for key, value in experiment_info['params'].items():
                    if isinstance(value, bool):
                        if value is True:
                            cmd_params.append(f"--{key} {str(value)}")
                    elif isinstance(value, list) and value:
                        list_items = []
                        for item in value:
                            item_str = str(item)
                            if ' ' in item_str or '=' in item_str or '"' in item_str:
                                list_items.append(f"'{item_str}'")
                            else:
                                list_items.append(item_str)
                        cmd_params.append(f"--{key} {' '.join(list_items)}")
                    else:
                        if value is None:
                            continue
                        elif isinstance(value, str) and (' ' in value or '=' in value or '"' in value):
                            cmd_params.append(f"--{key} '{value}'")
                        else:
                            cmd_params.append(f"--{key} {value}")
                full_cmd = f"python {os.path.basename(sys.argv[0])} {' '.join(cmd_params)}"
                f.write(full_cmd)
                f.write("\n\n")
            if experiment_info:
                f.write("===== Experiment Parameters =====\n")
                basic_info_keys = ['dataset', 'model_name', 'task_type', 'split_method', 'seed', 'timestamp', 'device']
                for key in basic_info_keys:
                    if key in experiment_info:
                        f.write(f"{key}: {experiment_info[key]}\n")
                if 'params' in experiment_info:
                    f.write("\nDetailed Parameters:\n")
                    for key, value in experiment_info['params'].items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            f.write("===== Detailed Split Results =====\n")
            for result in results_to_use:
                f.write(f"\nSplit {result['split_idx']+1}:")
                if 'test_subject' in result:
                    f.write(f" Subject {result['test_subject']}")
                elif 'fold_idx' in result:
                    f.write(f" Fold {result['fold_idx']}")
                f.write("\n")
                score_value = result.get('best_val_score', 0)
                f.write(f"Best validation score: {score_value:.4f}\n")
                if 'final_loss' in result and result['final_loss'] is not None:
                    f.write(f"Final loss: {result['final_loss']:.4f}\n")
                if 'final_train_metrics' in result and isinstance(result['final_train_metrics'], dict):
                    f.write("\nTraining metrics:\n")
                    self._write_metrics_section(f, result['final_train_metrics'], task_type)
                if 'final_metrics' in result and isinstance(result['final_metrics'], dict):
                    f.write("\nTest metrics:\n")
                    self._write_metrics_section(f, result['final_metrics'], task_type)
            f.write("\n===== Experiment Summary =====\n")
            f.write(f"Number of splits: {len(results_to_use)}\n")
            f.write(f"Task type: {task_type}\n")
            if avg_metrics:
                if 'avg_best_val_score' in avg_metrics:
                    f.write(f"Average best validation score: {avg_metrics['avg_best_val_score']:.4f}\n")
                train_metrics = {k: v for k, v in avg_metrics.items() if k.startswith('train_') and not k.endswith('_std')}
                if train_metrics:
                    f.write("\nTraining set average metrics:\n")
                    metrics_map = {k[6:]: v for k, v in train_metrics.items()}
                    ordered_metrics = []
                    if results_to_use and 'final_train_metrics' in results_to_use[0] and isinstance(results_to_use[0]['final_train_metrics'], dict):
                        for metric in results_to_use[0]['final_train_metrics']:
                            if metric in metrics_map:
                                ordered_metrics.append((metric, metrics_map[metric]))
                    for metric, value in metrics_map.items():
                        if metric not in [m[0] for m in ordered_metrics]:
                            ordered_metrics.append((metric, value))
                    # Write all metrics
                    for metric, value in ordered_metrics:
                        std_key = f"train_{metric}_std"
                        if std_key in avg_metrics:
                            f.write(f"  {metric}: {value:.4f} ± {avg_metrics[std_key]:.4f}\n")
                        else:
                            f.write(f"  {metric}: {value:.4f}\n")
            if avg_metrics:
                test_metrics = {k: v for k, v in avg_metrics.items() if k not in ['avg_best_val_score'] and not k.startswith('train_') and not k.endswith('_std')}
                if test_metrics:
                    f.write("\nTest set average metrics:\n")
                    ordered_metrics = []
                    if results_to_use and 'final_metrics' in results_to_use[0] and isinstance(results_to_use[0]['final_metrics'], dict):
                        for metric in results_to_use[0]['final_metrics']:
                            if metric in test_metrics:
                                ordered_metrics.append((metric, test_metrics[metric]))
                    for metric, value in test_metrics.items():
                        if metric not in [m[0] for m in ordered_metrics]:
                            ordered_metrics.append((metric, value))
                    # Write all metrics
                    for metric, value in ordered_metrics:
                        std_key = f"{metric}_std"
                        if std_key in avg_metrics:
                            f.write(f"  {metric}: {value:.4f} ± {avg_metrics[std_key]:.4f}\n")
                        else:
                            f.write(f"  {metric}: {value:.4f}\n")
            
            self.log(f'Results text file saved to {file_path}')
    
    def save_results_as_csv(self, file_path=None, filename='results.csv', split_results=None, all_seed_results=None, experiment_info=None):
        """Save hyperparameters and accuracy/metric matrices to CSV; supports single-seed or multi-seed aggregation."""
        if not file_path:
            if not self.experiment_dir:
                self.log("Warning: No output directory specified, results CSV not saved.")
                return
            output_dir = self.experiment_dir
        else:
            output_dir = os.path.dirname(file_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        dataset_name = experiment_info.get('dataset', 'unknown_dataset') if experiment_info else 'unknown_dataset'
        model_name = experiment_info.get('model_name', 'unknown_model') if experiment_info else 'unknown_model'
        
        hyperparams_csv_path = os.path.join(output_dir, f"hyperparams_{dataset_name}_{model_name}.csv")
        self._save_hyperparameters_to_csv(hyperparams_csv_path, experiment_info)
        
        if all_seed_results:
            self._save_accuracy_matrix_multiple_seeds_to_csv(output_dir, all_seed_results, experiment_info)
        else:
            results_to_use = split_results if split_results is not None else self.split_results
            if not results_to_use:
                self.log("Warning: No split results available to save as CSV.")
                return
            accuracy_path_prefix = os.path.join(output_dir, f"accuracy_{dataset_name}_{model_name}")
            self._save_accuracy_matrix_to_csv(accuracy_path_prefix, results_to_use)
        
        self.log(f'Results CSV files saved to {output_dir}')
    
    def manage_timepoint_directories(self, dataset_name, model_name, task_mode):
        """Keep the latest timepoint dirs under outputs/logs/{dataset}/{model}/{task_mode}, remove older ones (max 25)."""
        base_dir = os.path.join('./outputs', 'logs', dataset_name, model_name, task_mode)
        if not os.path.exists(base_dir):
            return
        
        timepoint_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and len(item) == 15 and '_' in item:
                parts = item.split('_')
                if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 8 and parts[1].isdigit() and len(parts[1]) == 6:
                    timepoint_dirs.append((item, os.path.getmtime(item_path)))
        
        timepoint_dirs.sort(key=lambda x: x[1], reverse=True)
        MAX_KEEP_TIMEPOINT_DIRS = 25
        for dir_name, _ in timepoint_dirs[MAX_KEEP_TIMEPOINT_DIRS:]:
            dir_path = os.path.join(base_dir, dir_name)
            shutil.rmtree(dir_path)
            self.log(f'Deleted old timepoint directory: {dir_path}')
    
    def _calculate_avg_metrics(self, results_to_use):
        """Compute mean and std of metrics across splits (classification or regression)."""
        if not results_to_use:
            return None
        
        avg_metrics = {}
        
        best_val_scores = []
        for r in results_to_use:
            if 'best_val_score' in r and r['best_val_score'] is not None:
                if hasattr(r['best_val_score'], 'dtype') and np.isscalar(r['best_val_score']):
                    best_val_scores.append(float(r['best_val_score']))
                elif isinstance(r['best_val_score'], (int, float)):
                    best_val_scores.append(r['best_val_score'])
        
        if best_val_scores:
            avg_metrics['avg_best_val_score'] = np.mean(best_val_scores)
        
        for result_type, result_key, prefix in [
            ('test', 'final_metrics', ''),
            ('train', 'final_train_metrics', 'train_')
        ]:
            all_metrics = set()
            for result in results_to_use:
                if result_key in result and isinstance(result[result_key], dict):
                    all_metrics.update(result[result_key].keys())
            
            for metric in all_metrics:
                values = []
                for result in results_to_use:
                    if (result_key in result and isinstance(result[result_key], dict) and 
                        metric in result[result_key]):
                        value = result[result_key][metric]
                        if hasattr(value, 'dtype') and np.isscalar(value):
                            values.append(float(value))
                        elif isinstance(value, (int, float)):
                            values.append(value)
                
                if values:
                    avg_metrics[f'{prefix}{metric}'] = np.mean(values)
                    avg_metrics[f'{prefix}{metric}_std'] = np.std(values)
        
        return avg_metrics
    
    def _write_metrics_section(self, f, metrics_dict, task_type):
        """Write metrics dict to file f as '  metric: value' lines."""
        if not isinstance(metrics_dict, dict):
            return
        for metric, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                f.write(f"  {metric}: {value:.4f}\n")
            elif hasattr(value, 'dtype') and np.isscalar(value):
                float_value = float(value)
                f.write(f"  {metric}: {float_value:.4f}\n")
    
    def _save_hyperparameters_to_csv(self, file_path, experiment_info):
        """Write experiment_info (basic + params) to CSV with columns 'Parameter', 'Value'."""
        hyperparams_data = []
        if experiment_info:
            basic_info_keys = ['dataset', 'model_name', 'task_type', 'split_method', 'seed', 'timestamp', 'device']
            for key in basic_info_keys:
                if key in experiment_info:
                    hyperparams_data.append({'Parameter': key, 'Value': experiment_info[key]})
            if 'params' in experiment_info:
                for key, value in experiment_info['params'].items():
                    hyperparams_data.append({'Parameter': key, 'Value': value})
        df = pd.DataFrame(hyperparams_data)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    def _save_accuracy_matrix_to_csv(self, output_path_prefix, split_results):
        """Save one CSV per metric: {output_path_prefix}_{metric}.csv; one row per subject, last row Overall average."""
        all_metrics = set()
        for split in split_results:
            metrics = split.get('final_metrics', {})
            all_metrics.update(metrics.keys())
        metrics_list = sorted(list(all_metrics))
        metrics_data = {metric: [] for metric in metrics_list}
        
        for split in split_results:
            # Get test subject or split index
            # Check whether test_subjects exists for n-fold cases
            test_subjects = split.get('test_subjects', None)
            if test_subjects is not None:
                # Convert the test_subjects array to a comma-separated string
                test_subject = ', '.join(map(str, test_subjects))
            else:
                # LOSO or other split cases
                test_subject = split.get('test_subject', split.get('fold_idx', f"Split {len(metrics_data[metrics_list[0]])+1}"))
            
            # Get metric data
            metrics = split.get('final_metrics', {})
            
            for metric in metrics_list:
                metric_value = metrics.get(metric, 0)
                if isinstance(metric_value, (int, float)):
                    formatted_value = self._format_metric_value(metric, metric_value)
                else:
                    formatted_value = str(metric_value)
                metrics_data[metric].append({
                    'Subject': test_subject,
                    f'{metric}': formatted_value
                })
        
        for metric in metrics_list:
            metric_file_path = f"{output_path_prefix}_{metric}.csv"
            df = pd.DataFrame(metrics_data[metric])
            
            avg_value = sum(float(item[metric]) for item in metrics_data[metric]) / len(metrics_data[metric])
            df.loc[len(df)] = ['Overall', f"{avg_value:.2f}"]
            df.to_csv(metric_file_path, index=False, encoding='utf-8-sig')

    def _save_accuracy_matrix_multiple_seeds_to_csv(self, output_dir, all_seed_results, experiment_info=None):
        """Save per-metric CSV: one row per seed, one column per subject, last row Subject_Avg."""
        dataset_name = experiment_info.get('dataset', 'unknown_dataset') if experiment_info else 'unknown_dataset'
        model_name = experiment_info.get('model_name', 'unknown_model') if experiment_info else 'unknown_model'
        all_metrics = set()
        for seed_result in all_seed_results:
            for split in seed_result.get('split_results', []):
                metrics = split.get('final_metrics', {})
                all_metrics.update(metrics.keys())
        metrics_list = sorted(list(all_metrics))
        seed_subject_data = {metric: {} for metric in metrics_list}
        all_subjects = set()
        seed_list = []
        for seed_result in all_seed_results:
            seed = seed_result.get('seed', f"Seed_{len(seed_list)}")
            seed_list.append(seed)
            for metric in metrics_list:
                seed_subject_data[metric][seed] = {}
            for split in seed_result.get('split_results', []):
                test_subject = split.get('test_subject', None)
                if test_subject is None:
                    test_subject = split.get('fold_idx', None)
                all_subjects.add(test_subject)
                metrics = split.get('final_metrics', {})
                for metric in metrics_list:
                    metric_value = metrics.get(metric, 0)
                    if isinstance(metric_value, (int, float)):
                        seed_subject_data[metric][seed][test_subject] = metric_value
                    else:
                        seed_subject_data[metric][seed][test_subject] = 0
        seed_list = sorted(seed_list)
        all_subjects = sorted(list(all_subjects))
        for metric in metrics_list:
            all_values, overall_avg = self._collect_metric_values(seed_subject_data, metric, seed_list, all_subjects, collect_all=True)
            metric_file_path = os.path.join(output_dir, f"{metric}_{overall_avg:.4f}_{dataset_name}_{model_name}.csv")
            data = []
            seed_avg_dict = self._collect_metric_values(seed_subject_data, metric, seed_list, all_subjects)
            for seed in seed_list:
                row = {'Seed': seed}
                for subject in all_subjects:
                    if subject in seed_subject_data[metric][seed]:
                        metric_value = seed_subject_data[metric][seed][subject]
                        formatted_value = self._format_metric_value(metric, metric_value)
                        row[f'Subject_{subject}'] = formatted_value
                    else:
                        row[f'Subject_{subject}'] = 'N/A'
                if seed_avg_dict[seed] is not None:
                    row['Seed_Avg'] = f"{seed_avg_dict[seed]:.{self._metric_decimals(metric)}f}"
                else:
                    row['Seed_Avg'] = 'N/A'
                row['Seed_Avg_Std'] = ''
                data.append(row)
            df = pd.DataFrame(data)
            subject_avg_row = {'Seed': 'Subject_Avg'}
            subject_seed_data = {metric: {}}
            for subject in all_subjects:
                subject_seed_data[metric][subject] = {}
                for seed in seed_list:
                    if subject in seed_subject_data[metric][seed]:
                        subject_seed_data[metric][subject][seed] = seed_subject_data[metric][seed][subject]
            subject_avg_dict = self._collect_metric_values(subject_seed_data, metric, all_subjects, seed_list)
            for subject in all_subjects:
                if subject_avg_dict[subject] is not None:
                    subject_avg_row[f'Subject_{subject}'] = f"{subject_avg_dict[subject]:.{self._metric_decimals(metric)}f}"
                else:
                    subject_avg_row[f'Subject_{subject}'] = 'N/A'
            subject_avg_row['Seed_Avg'] = f"{overall_avg:.{self._metric_decimals(metric)}f}" if all_values else 'N/A'
            seed_avg_values = [avg for avg in seed_avg_dict.values() if avg is not None]
            if seed_avg_values:
                import numpy as np
                seed_avg_std = np.std(seed_avg_values)
                subject_avg_row['Seed_Avg_Std'] = f"{seed_avg_std:.{self._metric_decimals(metric)}f}"
            else:
                subject_avg_row['Seed_Avg_Std'] = 'N/A'
            df.loc[len(df)] = subject_avg_row
            df.to_csv(metric_file_path, index=False, encoding='utf-8-sig')
            self.log(f"Metric {metric} results saved to {metric_file_path}")

    def _convert_metric_to_percentage(self, metric, value):
        """Convert accuracy-like metrics to percentage when value <= 1."""
        accuracy_metrics = ['balanced_accuracy', 'accuracy', 'cohen_kappa', 'f1_score', 'auroc', 'auprc']
        use_pct = metric in accuracy_metrics
        if not use_pct and metric.startswith('global_top'):
            use_pct = True
        if not use_pct and metric.startswith('v') and '_top' in metric:
            head, _, tail = metric.partition('_top')
            if len(head) > 1 and head[1:].isdigit() and tail.isdigit():
                use_pct = True
        if use_pct and value <= 1:
            return value * 100
        return value

    def _format_metric_value(self, metric, value):
        """Format metric value as string with metric-aware decimals."""
        converted_value = self._convert_metric_to_percentage(metric, value)
        return f"{converted_value:.{self._metric_decimals(metric)}f}"

    def _metric_decimals(self, metric):
        """Use 4 decimals for regression metrics, 2 for others."""
        regression_metrics = {'MSE', 'RMSE', 'MAE', 'R2', 'CORR', 'loss'}
        return 4 if metric in regression_metrics else 2

    def _collect_metric_values(self, seed_subject_data, metric, first_dimension, second_dimension, collect_all=False):
        """Collect metric values along dimensions; if collect_all True return (all_values, overall_avg) else return dict of first_dimension averages."""
        all_values = []
        dimension_avg_dict = {}
        
        for first_item in first_dimension:
            first_item_values = []
            
            for second_item in second_dimension:
                if second_item in seed_subject_data[metric][first_item]:
                    value = seed_subject_data[metric][first_item][second_item]
                    # Convert to percentage if needed
                    converted_value = self._convert_metric_to_percentage(metric, value)
                    first_item_values.append(converted_value)
                    if collect_all:
                        all_values.append(converted_value)
            if first_item_values:
                dimension_avg_dict[first_item] = sum(first_item_values) / len(first_item_values)
            else:
                dimension_avg_dict[first_item] = None
        if collect_all:
            overall_avg = sum(all_values) / len(all_values) if all_values else 0
            return all_values, overall_avg
        return dimension_avg_dict

    def _json_serializable(self, obj):
        """Convert non-JSON-serializable objects (ndarray, numpy scalars, datetime) to serializable form."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)

global_logger = Logger()

def get_logger():
    return global_logger

def set_global_logger(logger):
    global global_logger
    global_logger = logger