import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as data
from pytorch_lightning import loggers as pl_loggers
from train import train_model, test_model
import yaml
import traceback
from sklearn.model_selection import train_test_split
import math

from project_utils import *
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
test_mode = 0 if torch.cuda.is_available() else 1

# Load the config file
config_path = 'config.yml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Define dataset parameters directly in the script
DATASET_PATH = "data/"
#DATASET_NAME = "Caltech-256"  # Use "CIFAR-10" or "TinyImageNet" as needed
#DATASET_NAME = "CIFAR-10"  # Use "CIFAR-10" or "TinyImageNet" as needed
DATASET_NAME = "TinyImageNet"  # Use "CIFAR-10" or "TinyImageNet" as needed
MODEL_NAME = "ViT"  # Use "ResNet-18" as needed
#MODEL_NAME = "ResNet-18"  # Use "ResNet-18" as needed
SUBSET_PERCENTAGE = 0.01  # For example, 0.1 use 10% of the dataset
deterministic_seed = 42
run_name = f"{DATASET_NAME}_sample_ratio{SUBSET_PERCENTAGE}_{MODEL_NAME}_seed-{deterministic_seed}"
config['run_name'] = run_name
print("Run name: " + run_name)

LOGGER_PATH = "narvi_logs/"
LOGGER_NAME = "ViT_run"
CHECKPOINT_PATH = "saved_models/resnet18/"

if test_mode:
    SUBSET_PERCENTAGE = SUBSET_PERCENTAGE / 10
    config['TRAINER']['MAX_EPOCHS'] = 1
config['MODEL_ARGS']['MODEL_NAME'] = MODEL_NAME

if DATASET_NAME == "TinyImageNet":
    config['MODEL_ARGS']['num_classes'] = 200
    config['MODEL_ARGS']['patch_size'] = 8
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # Replace with calculated mean and std
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),

        # Replace with calculated mean and std
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
elif DATASET_NAME == "CIFAR-10":
    config['MODEL_ARGS']['num_classes'] = 10
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),

    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),

    ])
else:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}")


def get_balanced_train_subset(dataset, subset_percentage, seed=deterministic_seed):
    targets = dataset.dataset.targets if isinstance(dataset, torch.utils.data.Subset) else dataset.targets
    n_classes = len(set(targets))

    # Calculate the number of samples based on the subset percentage
    num_samples = int(len(targets) * subset_percentage)

    # Ensure we have at least as many samples as there are classes
    if num_samples < n_classes:
        print(
            f"Requested subset percentage results in fewer samples ({num_samples}) than the number of classes ({n_classes}). "
            f"Adjusting to use all classes with {n_classes} samples.")
        num_samples = n_classes
        subset_percentage = num_samples / len(targets)

    train_indices, _ = train_test_split(
        range(len(targets)), train_size=subset_percentage, stratify=targets, random_state=seed
    )
    return train_indices


def get_balanced_val_subset(dataset, subset_percentage, seed=deterministic_seed):
    targets = [dataset[i][1] for i in range(len(dataset))]
    n_classes = len(set(targets))

    # Calculate the number of samples based on the subset percentage
    num_samples = int(len(targets) * subset_percentage)

    # Ensure we have at least as many samples as there are classes
    if num_samples < n_classes:
        print(
            f"Requested subset percentage results in fewer samples ({num_samples}) than the number of classes ({n_classes}). "
            f"Adjusting to use all classes with {n_classes} samples.")
        num_samples = n_classes
        subset_percentage = num_samples / len(targets)

    val_indices, _ = train_test_split(
        range(len(targets)), train_size=subset_percentage, stratify=targets, random_state=seed
    )
    return val_indices
# Set up data transformations
train_transform = transforms.Compose([
#    transforms.Resize((224, 224)),  # Resize images to 224x224 for ViT
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
#    transforms.Resize((224, 224)),  # Resize images to 224x224 for ViT
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load dataset
train_dataset, val_dataset, test_dataset = load_dataset(DATASET_NAME, DATASET_PATH, train_transform, test_transform)

# Get balanced subset
subset_indices_train = get_balanced_train_subset(train_dataset, SUBSET_PERCENTAGE)
train_dataset = torch.utils.data.Subset(train_dataset, subset_indices_train)
subset_indices_val = get_balanced_val_subset(val_dataset, SUBSET_PERCENTAGE)
val_dataset = torch.utils.data.Subset(val_dataset, subset_indices_val)

# Create DataLoader with adaptive batch size
train_loader = data.DataLoader(
    train_dataset,
    batch_size=min(128, len(train_dataset)),
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=4
)
val_loader = data.DataLoader(
    val_dataset,
    batch_size=min(128, len(val_dataset)),
    shuffle=False,
    drop_last=False,
    num_workers=4
)
test_loader = data.DataLoader(
    test_dataset,
    batch_size=min(128, len(test_dataset)),
    shuffle=False,
    drop_last=False,
    num_workers=4
)

# Set up logger
logger = pl_loggers.TensorBoardLogger(LOGGER_PATH, name=LOGGER_NAME)

# Training loop with error handling
try:
    model, trainer = train_model(config, logger, train_loader, val_loader)
    trainer.logger.finalize("ok")
except KeyboardInterrupt:
    print("KeyboardInterrupt. Aborting.")
    raise
except Exception as err:
    print(type(err))  # the exception type
    print(err)  # actual error
    traceback.print_exc()

# Optional: Test the model after training
"""
try:
    test_model(model, test_loader, trainer)
except Exception as err:
    print("Testing failed.")
    print(type(err))  # the exception type
    print(err)  # actual error
    traceback.print_exc()
"""
