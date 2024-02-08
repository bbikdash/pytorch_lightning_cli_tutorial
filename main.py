import os
import sys
# sys.path.insert(1, "/home/bbikdash/Development/10_bassam_devel_utils/")
import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.cli import LightningCLI

# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
# import my_models.my_models  # noqa: F401
# import my_datasets.data_module  # noqa: F401

class Model1(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model1", "⚡")
        return super().configure_optimizers()


class Model2(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model2", "⚡")
        return super().configure_optimizers()

class FakeDataset1(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset1", "⚡")
        return torch.utils.data.DataLoader(self.random_train)


class FakeDataset2(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset2", "⚡")
        return torch.utils.data.DataLoader(self.random_train)

# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.link_arguments("optimizer.lr", "model.learning_rate", apply_on="instantiate")

def cli_main():
    
    # cli = MyLightningCLI()
    cli = LightningCLI()
    # note: don't call fit!!

    """
    Valid configurations:
    cli = LightningCLI(DemoModel, BoringDataModule)
    python main.py fit --trainer.max_epochs=100

    
    cli = LightningCLI()
    python main.py fit --config config.yaml --print_config
    python main.py fit --model Model1 --data FakeDataset2 --trainer.max_epochs=50
    python main.py fit --model Model1 --data FakeDataset1 --trainer.max_epochs=50 --optimizer AdamW --optimizer.lr=0.01

    """


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
