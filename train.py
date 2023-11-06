import os
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as data

import yaml
from model.ViT import ViT
from pytorch_lightning import loggers as pl_loggers
from get_device_info import get_cpu_name
from get_device_info import get_cuda_names
from datetime import datetime
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
def test_model(cfg, test_loader):
    trainer.test(model, dataloaders=test_loader, verbose=True)

def train_model(cfg, train_loader, val_loader):
    logger = pl_loggers.TensorBoardLogger(cfg["LOGGER"]["logger_path"], name=cfg["LOGGER"]["logger_name"])
    logger.log_hyperparams(
        params=dict(cpu_info=get_cpu_name(),
                    cuda_info=get_cuda_names(),
                    time_start_train=str(datetime.now().strftime('%d.%m.%Y %H:%M:%S')),
                    data_sampleset_name=cfg["DATASET"]["DATA_SAMPLESET_NAME"]

        )
    )
    logger.save()
    trainer = L.Trainer(
        default_root_dir=os.path.join(cfg["TRAINER"]["CHECKPOINT_PATH"], "ViT"),
        accelerator="auto",
        devices=1,
        max_epochs=cfg["TRAINER"]["MAX_EPOCHS"], #default 180
        logger=logger,
        log_every_n_steps=cfg["TRAINER"]["LOG_EVERY_N_STEPS"],
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
    )

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(42)  # To be reproducable
    model = ViT(cfg, train_loader)
    trainer.fit(model, train_loader, val_loader)
    # Load best checkpoint after training
    model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)


    return model, trainer

cfgs = ['configs/ViT_s1000_e3-1.yml',
        'configs/ViT_s1000_e4-1.yml',
        'configs/ViT.yml']
with open('configs/ViT.yml', 'r') as file:
    cfg = yaml.safe_load(file)
for cfgpath in cfgs:
    with open(cfgpath, 'r') as file:
        cfg = yaml.safe_load(file)

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),])

    train_dataset = CIFAR10(root=cfg["DATASET"]["DATASET_PATH"], train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=cfg["DATASET"]["DATASET_PATH"], train=True, transform=test_transform, download=True)
    test_dataset = CIFAR10(root=cfg["DATASET"]["DATASET_PATH"], train=False, transform=test_transform, download=True)
    L.seed_everything(42)
    with open(cfg["DATASET"]["DATA_SAMPLESET_NAME"], 'r') as file:
        sampleset = yaml.load(file, Loader=yaml.FullLoader)
    print(len(sampleset))
    test_set = test_dataset
    L.seed_everything(42)
    train_set = torch.utils.data.Subset(train_dataset, sampleset['Train_sample_ids'])
    val_set = torch.utils.data.Subset(train_dataset, sampleset['Val_sample_ids'])

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=cfg["TRAIN_LOADER"]["BATCH_SIZE"], shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=cfg["TRAIN_LOADER"]["BATCH_SIZE"], shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=cfg["TRAIN_LOADER"]["BATCH_SIZE"], shuffle=False, drop_last=False, num_workers=4)



    devices_cuda = [d for d in range(torch.cuda.device_count())]
    devices_cpu = [d for d in range(torch.cpu.device_count())]
    model, trainer = train_model(cfg, train_loader, val_loader)
    test_model(cfg,test_loader)
    trainer.logger.finalize("ok")
#   LOG THESE:?
#    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
#    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)


