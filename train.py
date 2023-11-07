# make trainer use all devices
# make conda environment & export it
# more subsets which are randomly generated
# list dependencies
# test before commiting for experiments


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
