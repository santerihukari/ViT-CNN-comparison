import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.interpolate import griddata


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
    for log_path in log_paths:
        if dataset_name in log_path and model_name in log_path:
            ratio = extract_ratio_from_path(log_path)
            if ratio is not None:
                filtered_logs.append((log_path, ratio))
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


def rolling_average(data, window_size):
    """Apply a rolling average with a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def normalize_epochs(epochs, new_min=1, new_max=180):
    """Normalize epoch values to the range [new_min, new_max]."""
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    return [new_min + (e - min_epoch) * (new_max - new_min) / (max_epoch - min_epoch) for e in epochs]


def plot_3d_surface(logs_data, dataset_name, model_name, metric_name, fig_num):
    fig = plt.figure(fig_num, figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    epoch_values = []
    ratio_values = []
    metric_values = []

    for log_path, ratio in logs_data:
        epochs, values = extract_metric_epoch(log_path, metric_name)
        if epochs is not None:
            # Normalize the epoch values to fit within [1, 180] range for each log individually
            norm_epochs = normalize_epochs(epochs)

            # Smooth the metric values using a 3-step rolling average
            smoothed_values = rolling_average(values, window_size=3)
            smoothed_epochs = norm_epochs[:len(smoothed_values)]  # Align epochs with smoothed values

            # Collect all data points
            epoch_values.extend(smoothed_epochs)
            ratio_values.extend([ratio] * len(smoothed_values))
            metric_values.extend(smoothed_values)

    # Create a grid for the surface plot
    epoch_grid, ratio_grid = np.meshgrid(
        np.linspace(min(epoch_values), max(epoch_values), num=50),
        np.linspace(min(ratio_values), max(ratio_values), num=50)
    )

    # Interpolate metric values on the grid
    grid_metric = griddata(
        (epoch_values, ratio_values),
        metric_values,
        (epoch_grid, ratio_grid),
        method='cubic'
    )

    # Plot surface
    surf = ax.plot_surface(epoch_grid, ratio_grid, grid_metric, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Normalized Epoch')
    ax.set_ylabel('Dataset Ratio')
    ax.set_zlabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'3D Surface Plot of {metric_name.replace("_", " ").title()} ({dataset_name}/{model_name})')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Plot solid black lines at epochs 100 and 150
    for epoch in [100, 150]:
        # Find the index for the given epoch value
        epoch_index = np.abs(epoch_grid[0, :] - epoch).argmin()
        ax.plot(
            [epoch] * len(ratio_grid[:, 0]),
            ratio_grid[:, 0],
            grid_metric[:, epoch_index],
            color='black',
            linestyle='-',
            linewidth=2,  # Adjusted thickness
        )
        ax.text(
            epoch,
            ratio_grid[0, 0],
            max(grid_metric[:, epoch_index]),
            f'Epoch {epoch}',
            color='black',
            fontsize=9,  # Smaller font size
            weight='bold'
        )


def main():
    base_dir = '/home/santeri/ViT-CNN-comparison/TINY-DEBUG_06-08-2024_testruns'
    metric_name = 'train_acc_epoch'  # Change this to the metric you want to compare

    datasets = ['TinyImageNet']
    models = ['ViT']
#    datasets = ['CIFAR-10', 'TinyImageNet']
#    models = ['ResNet-18', 'ViT']

    fig_num = 1  # Start figure numbering

    for dataset_name in datasets:
        for model_name in models:
            log_paths = list_and_sort_tensorboard_logs(base_dir)
            logs_data = filter_logs(log_paths, dataset_name, model_name)

            if not logs_data:
                print(f"No logs found for dataset {dataset_name} and model {model_name}.")
                continue

            print(f"Found {len(logs_data)} logs for dataset {dataset_name} and model {model_name}.")
            plot_3d_surface(logs_data, dataset_name, model_name, metric_name, fig_num)
            fig_num += 1  # Increment figure number for each plot

    plt.show()  # Show all plots at once


if __name__ == "__main__":
    main()
