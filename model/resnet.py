import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import urllib.request
from types import SimpleNamespace
from torchmetrics.classification import Precision, Recall, F1Score

import lightning as L
import torch.nn as nn

from torchvision.models import resnet18


model_dict = {}
act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}


class RN(L.LightningModule):
    def __init__(self, cfg, train_loader):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.lr = cfg["MODEL_ARGS"]["learning_rate"]
        model_kwargs = {
            "num_classes": cfg["MODEL_ARGS"]["num_classes"],
        }
        self.model = resnet18(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_loss" % mode, loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("%s_acc" % mode, acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
