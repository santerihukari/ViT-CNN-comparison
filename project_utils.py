import os
import urllib.request
import tarfile
import zipfile
from torchvision.datasets import CIFAR10, ImageFolder
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Subset, Dataset
from PIL import Image
import pandas as pd


class TinyImageNetValidationDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_labels = pd.read_csv(annotations_file, sep='\t', header=None,
                                        names=['filename', 'class', 'column2', 'column3', 'column4', 'column5'])
        # Ensure filenames are treated as strings
        self.image_labels['filename'] = self.image_labels['filename'].astype(str)

        # Create file paths
        self.image_labels['filepath'] = self.image_labels['filename'].apply(lambda x: os.path.join(images_dir, x))

        # Create class-to-index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.image_labels['class'].unique())}
        self.image_labels['class_idx'] = self.image_labels['class'].apply(lambda x: self.class_to_idx[x])

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
#        print(self.image_labels[1:10])
        img_path = self.image_labels.iloc[idx]['filepath']
#        print("img_path:",img_path)
        img = Image.open(img_path).convert('RGB')
        label = self.image_labels.iloc[idx]['class_idx']
        if self.transform:
            img = self.transform(img)
        return img, label

def get_cpu_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""
def get_cuda_names():
    devices = [d for d in range(torch.cuda.device_count())]
    device_names = [torch.cuda.get_device_name(d) for d in devices]
    return device_names
def download_and_extract_caltech256(url, download_path, extract_path):
    """Download and extract Caltech-256 dataset."""
    if not os.path.exists(extract_path):
        print(f"Downloading Caltech-256 from {url} to {download_path}...")
        urllib.request.urlretrieve(url, download_path)
        print("Download complete.")

        print(f"Extracting {download_path} to {extract_path}...")
        with tarfile.open(download_path, 'r') as tar_ref:
            tar_ref.extractall(extract_path)
        print("Extraction complete.")
    else:
        print("Caltech-256 already downloaded and extracted.")


def download_and_extract_tinyimagenet(url, download_path, extract_path):
    """Download and extract Tiny ImageNet dataset."""
    if not os.path.exists(extract_path):
        print(f"Downloading Tiny ImageNet from {url} to {download_path}...")
        urllib.request.urlretrieve(url, download_path)
        print("Download complete.")

        print(f"Extracting {download_path} to {extract_path}...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    else:
        print("Tiny ImageNet already downloaded and extracted.")

def load_dataset(dataset_name, data_path, train_transform, test_transform, val_split=0.2, test_split=0.1, seed=42):
    """Load the dataset and return data loaders."""
    if dataset_name == "CIFAR-10":
        train_dataset = CIFAR10(root=data_path, train=True, transform=train_transform, download=True)
        val_dataset = CIFAR10(root=data_path, train=False, transform=test_transform, download=True)
        test_dataset = CIFAR10(root=data_path, train=False, transform=test_transform, download=True)
    elif dataset_name == "TinyImageNet":
        # Download and extract Tiny ImageNet
        tinyimagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        download_path = os.path.join(data_path, 'tiny-imagenet-200.zip')
        extract_path = os.path.join(data_path, 'tiny-imagenet-200')
        download_and_extract_tinyimagenet(tinyimagenet_url, download_path, extract_path)

        # Verify the directory structure
        train_dir = os.path.join(extract_path, 'tiny-imagenet-200/train')
        val_images_dir = os.path.join(extract_path, 'tiny-imagenet-200/val', 'images')
        val_annotations_file = os.path.join(extract_path, 'tiny-imagenet-200/val', 'val_annotations.txt')

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.isdir(val_images_dir):
            raise FileNotFoundError(f"Validation images directory not found: {val_images_dir}")
        if not os.path.isfile(val_annotations_file):
            raise FileNotFoundError(f"Validation annotations file not found: {val_annotations_file}")

        # Load Tiny ImageNet
        # Training dataset
        train_dataset = ImageFolder(root=train_dir, transform=train_transform)

        # Validation dataset
        val_dataset = TinyImageNetValidationDataset(val_images_dir, val_annotations_file, transform=test_transform)

        # There is no predefined test dataset, so we'll use a subset of validation set as the test set
        num_val_samples = len(val_dataset)
        val_indices, test_indices = train_test_split(
            list(range(num_val_samples)), test_size=test_split, random_state=seed
        )
        val_dataset = Subset(val_dataset, val_indices)
        test_dataset = Subset(val_dataset, test_indices)
    elif dataset_name == "Caltech-256":
        # Download and extract Caltech-256
        caltech256_url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
        download_path = os.path.join(data_path, '256_ObjectCategories.tar')
        extract_path = os.path.join(data_path, '256_ObjectCategories')
        download_and_extract_caltech256(caltech256_url, download_path, extract_path)

        # Load Caltech-256 and create splits
        full_dataset = ImageFolder(root=extract_path, transform=train_transform)
        targets = full_dataset.targets

        # Split dataset into train, validation, and test sets
        train_indices, temp_indices = train_test_split(
            range(len(full_dataset)), test_size=val_split + test_split, random_state=seed, stratify=targets
        )
        val_size = val_split / (val_split + test_split)
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=1 - val_size, random_state=seed, stratify=[targets[i] for i in temp_indices]
        )

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    return train_dataset, val_dataset, test_dataset