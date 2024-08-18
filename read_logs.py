import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator





def list_and_sort_tensorboard_logs(base_dir):
    log_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                log_paths.append(os.path.join(root, file))

    # Sort logs by dataset ratio extracted from directory names
    log_paths.sort(key=lambda x: extract_ratio_from_path(x))
    return log_paths


def extract_ratio_from_path(path):
    match = re.search(r'_ratio(\d+(\.\d+)?)_', path)
    if match:
        return float(match.group(1))
    return None


def filter_logs(log_paths, dataset_name, model_name):
    filtered_logs = []
    seen_versions = set()

    for log_path in log_paths:
        if dataset_name in log_path and model_name in log_path and 'tcsc' in log_path:
            ratio = extract_ratio_from_path(log_path)
            # Extract unique identifier for the log, typically the version directory
            version = os.path.dirname(log_path).split('/')[-1]
            identifier = (ratio, version)

            if ratio is not None and identifier not in seen_versions:
                filtered_logs.append((log_path, ratio))
                seen_versions.add(identifier)

    return filtered_logs


def extract_metric_epoch(log_path, metric_name):
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()

    scalar_tags = event_acc.Tags().get('scalars', [])
    if metric_name not in scalar_tags:
        print(f"'{metric_name}' tag not found in {log_path}")
        return None, None

    events = event_acc.Scalars(metric_name)
    epochs = [event.step for event in events]
    values = [event.value for event in events]
    return epochs, values


def normalize_epochs(epochs, new_min=1, new_max=180):
    """Normalize epoch values to the range [new_min, new_max]."""
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    return [new_min + (e - min_epoch) * (new_max - new_min) / (max_epoch - min_epoch) for e in epochs]


def plot_2d_accuracy_vs_epoch(logs_data, dataset_name, model_name, metric_name, output_dir):
    plt.figure(figsize=(10, 6))

    for log_path, ratio in logs_data:
        epochs, values = extract_metric_epoch(log_path, metric_name)
        if epochs is not None:
            # Normalize the epoch values to fit within [1, 180] range
            norm_epochs = normalize_epochs(epochs)
            plt.plot(norm_epochs, values, label=f'Ratio {ratio:.3f}')

    # Add vertical lines at epochs 100 and 150 to indicate learning rate reductions
    plt.axvline(x=100, color='black', linestyle='--', linewidth=2, label='LR reduced')
    plt.axvline(x=150, color='black', linestyle='--', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} vs Epoch ({dataset_name}/{model_name})')
    plt.legend()
    plt.tight_layout()

    file_name = f'{output_dir}/{dataset_name}_{model_name}_{metric_name}_accuracy_vs_epoch.png'
    plt.savefig(file_name)
    plt.close()


def plot_2d_best_accuracy_vs_ratio(logs_data, dataset_name, model_name, metric_name, output_dir):
    ratios = []
    best_accuracies = []

    for log_path, ratio in logs_data:
        epochs, values = extract_metric_epoch(log_path, metric_name)
        if epochs is not None:
            best_accuracy = max(values)
            ratios.append(ratio)
            best_accuracies.append(best_accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(ratios, best_accuracies, marker='o')
    plt.xlabel('Subset Ratio')
    plt.ylabel(f'Best {metric_name.replace("_", " ").title()}')
    plt.title(f'Best {metric_name.replace("_", " ").title()} vs Subset Ratio ({dataset_name}/{model_name})')
    plt.tight_layout()

    file_name = f'{output_dir}/{dataset_name}_{model_name}_{metric_name}_best_accuracy_vs_ratio.png'
    plt.savefig(file_name)
    plt.close()


def save_2d_plots(dataset_name, model_name):
    metrics = ['val_acc_epoch', 'val_loss_epoch', 'train_acc_epoch', 'train_loss_epoch']

    # Create directory if it does not exist
    output_dir = 'imgs'
    os.makedirs(output_dir, exist_ok=True)

    base_dir = '/home/santeri/ViT-CNN-comparison/TINY-DEBUG_06-08-2024_testruns'
    log_paths = list_and_sort_tensorboard_logs(base_dir)
    logs_data = filter_logs(log_paths, dataset_name, model_name)

    if not logs_data:
        print(f"No logs found for dataset {dataset_name} and model {model_name}.")
        return

    print(f"Found {len(logs_data)} logs for dataset {dataset_name} and model {model_name}.")

    for metric_name in metrics:
        plot_2d_accuracy_vs_epoch(logs_data, dataset_name, model_name, metric_name, output_dir)
        plot_2d_best_accuracy_vs_ratio(logs_data, dataset_name, model_name, metric_name, output_dir)


def main():
    # Instructions for different types of plots:

    # 1. 3D Surface Plot:
    # To generate a 3D surface plot, call the function `plot_3d_surface()`.
    # Example:
    # plot_3d_surface(dataset_name='CIFAR-10', model_name='ResNet-18', metric_name='val_acc_epoch')

    # 2. 3D Individual Logs Plot:
    # To generate a 3D plot with individual logs without connecting them to a plane, call `plot_3d_individual_logs()`.
    # Example:
    # plot_3d_individual_logs(dataset_name='TinyImageNet', model_name='ViT', metric_name='val_acc_epoch')

    # 3. 2D Accuracy vs Epoch:
    # To generate a 2D plot for accuracy vs. epoch for all subset ratios, call `plot_2d_accuracy_vs_epoch()`.
    # Example:
    # plot_2d_accuracy_vs_epoch(logs_data, 'CIFAR-10', 'ResNet-18', 'val_acc_epoch', output_dir)

    # 4. 2D Best Accuracy vs Subset Ratio:
    # To generate a 2D plot for best accuracy vs. subset ratio, call `plot_2d_best_accuracy_vs_ratio()`.
    # Example:
    # plot_2d_best_accuracy_vs_ratio(logs_data, 'TinyImageNet', 'ViT', 'val_acc_epoch', output_dir)

    # For now, we're saving all 2D plots for CIFAR-10/ResNet-18:
    dataset_name = 'TinyImageNet'  # Change this to the dataset you want
    model_name = 'ViT'  # Change this to the model you want

    save_2d_plots(dataset_name, model_name)


if __name__ == "__main__":
    main()
