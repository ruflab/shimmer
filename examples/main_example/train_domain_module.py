import os
import shutil
from typing import Literal

from dataset import DomainDataModule, domain_sizes, get_domain_data
from domains import GenericDomain
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


def train_unimodal_module(module_name: Literal["domain1", "domain2"]):
    train_data, val_data = get_domain_data(module_name)

    data = DomainDataModule(train_data, val_data, batch_size=32)
    domain_module = GenericDomain(input_size=domain_sizes[module_name], latent_dim=32)

    trainer = Trainer(
        max_epochs=4,
        log_every_n_steps=4,  # log metrics every 4 training steps
        devices=1,  # use only one GPU when training
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename=module_name,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ],
    )
    trainer.fit(domain_module, data)
    trainer.validate(domain_module, data, "best")


if __name__ == "__main__":
    # reset the checkpoints directory
    shutil.rmtree("checkpoints")
    os.mkdir("checkpoints")

    # Let's train domain1 and domain2
    train_unimodal_module("domain1")
    train_unimodal_module("domain2")
