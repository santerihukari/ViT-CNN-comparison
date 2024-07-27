import os
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from model.ViT import ViT
from model.resnet import RN
from pytorch_lightning import loggers as pl_loggers
from get_device_info import get_cpu_name, get_cuda_names
from datetime import datetime

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

def test_model(model, test_loader, trainer):
    trainer.test(model, dataloaders=test_loader, verbose=True)

def train_model(cfg, model_logger, train_loader, val_loader):
    logger = model_logger
    print("Logging hyperparameters")
    logger.log_hyperparams(
        params=dict(run_name=cfg['run_name'],
                    num_train_samples=len(train_loader.dataset),  # Log number of training samples
                    num_val_samples=len(val_loader.dataset),  # Log number of validation samples
                    cpu_info=get_cpu_name(),
                    cuda_info=get_cuda_names(),
                    time_start_train=str(datetime.now().strftime('%d.%m.%Y %H:%M:%S'))
        )
    )
    logger.save()

    trainer = L.Trainer(
        default_root_dir=os.path.join(cfg["TRAINER"]["CHECKPOINT_PATH"], cfg["MODEL_ARGS"]["MODEL_NAME"]),
        accelerator="auto",
        devices=1 if (not torch.cuda.is_available() or cfg["TRAINER"]["TUNE"] == "True") else list(range(torch.cuda.device_count())),
        max_epochs=cfg["TRAINER"]["MAX_EPOCHS"],
        logger=logger,
        log_every_n_steps=cfg["TRAINER"]["LOG_EVERY_N_STEPS"],
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
    )

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    L.seed_everything(42)  # Ensure reproducibility

    if cfg["MODEL_ARGS"]["MODEL_NAME"] == "ResNet-18":
        model = RN(cfg, train_loader)
        trainer.fit(model, train_loader, val_loader)
        model = RN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    elif cfg["MODEL_ARGS"]["MODEL_NAME"] == "ViT":
        model = ViT(cfg, train_loader)
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)


    else:
        raise ValueError("Unsupported model type in configuration.")
    return model, trainer