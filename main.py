import lightning as L
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as data
from train import *
import yaml
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


cfgs_vit_2 = [
    'configs_second_narvi_run/resnet18_narvi_1.yml',
    'configs_second_narvi_run/ViT_narvi_1.yml',
    'configs_second_narvi_run/resnet18_narvi_2.yml',
    'configs_second_narvi_run/ViT_narvi_2.yml',
    'configs_second_narvi_run/resnet18_narvi_3.yml',
    'configs_second_narvi_run/ViT_narvi_3.yml',
    'configs_second_narvi_run/resnet18_narvi_4.yml',
    'configs_second_narvi_run/ViT_narvi_4.yml',
    'configs_second_narvi_run/resnet18_narvi_5.yml',
    'configs_second_narvi_run/ViT_narvi_5.yml',
    'configs_second_narvi_run/resnet18_narvi_6.yml',
    'configs_second_narvi_run/ViT_narvi_6.yml',
    'configs_second_narvi_run/resnet18_narvi_7.yml',
    'configs_second_narvi_run/ViT_narvi_7.yml',
    'configs_second_narvi_run/resnet18_narvi_8.yml',
    'configs_second_narvi_run/ViT_narvi_8.yml',
    'configs_second_narvi_run/resnet18_narvi_9.yml',
    'configs_second_narvi_run/ViT_narvi_9.yml',
    'configs_second_narvi_run/resnet18_narvi_10.yml',
    'configs_second_narvi_run/ViT_narvi_10.yml'
    ]


cfgs = cfgs_vit_2

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                                                  [0.24703223, 0.24348513,
                                                                                   0.26158784]), ])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                                                 [0.24703223, 0.24348513,
                                                                                  0.26158784]), ])

for cfgpath in cfgs:
    with open(cfgpath, 'r') as file:
        cfg = yaml.safe_load(file)
    cfg["TRAINER"]["MAX_EPOCHS"] = cfg["TRAINER"]["MAX_EPOCHS"] if torch.cuda.is_available() else 5
    print("Using ",cfg["TRAINER"]["MAX_EPOCHS"]," epochs.")
    cifar_train = CIFAR10(root=cfg["DATASET"]["DATASET_PATH"], train=True, transform=train_transform, download=True)
    train_dataset = cifar_train
    with open(cfg["DATASET"]["DATA_SAMPLESET_NAME"], 'r') as file:
        sampleset = yaml.load(file, Loader=yaml.FullLoader)
    train_set = torch.utils.data.Subset(train_dataset, sampleset['Train_sample_ids'])
    val_set = torch.utils.data.Subset(train_dataset, sampleset['Val_sample_ids'])

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=cfg["TRAIN_LOADER"]["BATCH_SIZE"], shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=cfg["TRAIN_LOADER"]["BATCH_SIZE"], shuffle=False, drop_last=False, num_workers=4)

    model, trainer = train_model(cfg, train_loader, val_loader)
    trainer.logger.finalize("ok")
