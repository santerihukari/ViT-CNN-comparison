# make trainer use all devices // DONE
# make conda environment & export it // Created, just export it now
# more subsets which are randomly generated
# list dependencies // Partly done
# test before committing for experiments


import os
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from model.ViT import ViT
from model.resnet import RN
from pytorch_lightning import loggers as pl_loggers
from get_device_info import get_cpu_name
from get_device_info import get_cuda_names
from datetime import datetime
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
def test_model(cfg, test_loader):
    trainer.test(model, dataloaders=test_loader, verbose=True)

def train_model(cfg, model_logger, train_loader, val_loader):
    logger = model_logger
    logger.log_hyperparams(
        params=dict(cpu_info=get_cpu_name(),
                    cuda_info=get_cuda_names(),
                    time_start_train=str(datetime.now().strftime('%d.%m.%Y %H:%M:%S')),
                    data_sampleset_name=cfg["DATASET"]["DATA_SAMPLESET_NAME"]

        )
    )
    logger.save()

    trainer = L.Trainer(
        default_root_dir=os.path.join(cfg["TRAINER"]["CHECKPOINT_PATH"], cfg["MODEL_ARGS"]["MODEL_NAME"]),
        accelerator="auto",
        devices=1 if ((not torch.cuda.is_available()) or cfg["TRAINER"]["TUNE"] == "True") else list(range(torch.cuda.device_count())),
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

    L.seed_everything(42)  # To be reproducible
#    model = ViT(cfg, train_loader)
    if cfg["MODEL_ARGS"]["MODEL_NAME"] == "resnet18":
        model = RN(cfg, train_loader)
        trainer.fit(model, train_loader, val_loader)
        model = RN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)


    elif cfg["MODEL_ARGS"]["MODEL_NAME"] == "ViT":
        model = ViT(cfg, train_loader)
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Load best checkpoint after training

    return model, trainer

