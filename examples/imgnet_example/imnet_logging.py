import io
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import wandb

import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import gridspec
from matplotlib.figure import Figure
from PIL import Image
from shimmer import (
    SingleDomainSelection,
    batch_cycles,
    batch_demi_cycles,
    batch_translations,
)
from shimmer.modules.global_workspace import GlobalWorkspaceBase
from torchvision.utils import make_grid

import os
from torchvision.utils import make_grid, save_image

from sentence_transformers import SentenceTransformer

matplotlib.use("Agg")


def get_pil_image(figure: Figure) -> Image.Image:
    buf = io.BytesIO()
    figure.savefig(buf)
    buf.seek(0)
    return Image.open(buf)

class LogGWImagesCallback(pl.Callback):
    def __init__(
        self,
        reference_samples: Mapping[frozenset[str], Mapping[str, Any]],
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
        image_size: int = 32,
        ncols: int = 8,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.reference_samples = reference_samples
        self.every_n_epochs = every_n_epochs
        self.log_key = log_key
        self.image_size = image_size
        self.ncols = ncols

        # Initialize the text encoder model
        self.bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def to(
        self,
        samples: Mapping[frozenset[str], Mapping[str, Any]],
        device: torch.device,
    ) -> dict[frozenset[str], dict[str, Any]]:
        out: dict[frozenset[str], dict[str, Any]] = {}
        for domain_names, domains in samples.items():
            latents: dict[str, Any] = {}
            for domain_name, domain in domains.items():
                if isinstance(domain, torch.Tensor):
                    latents[domain_name] = domain.to(device)
                elif isinstance(domain, list) and all(isinstance(item, str) for item in domain):
                    # Keep the list of strings as is
                    latents[domain_name] = domain
                else:
                    latents[domain_name] = [x.to(device) for x in domain]
            out[domain_names] = latents
        return out


    def on_callback(
        self,
        current_epoch: int,
        loggers: Sequence[Logger],
        pl_module: GlobalWorkspaceBase,
        trainer: pl.Trainer,
    ) -> None:
        samples = self.to(self.reference_samples, pl_module.device)

        if current_epoch == 0:
            for domain_names, domains in samples.items():
                for domain_name, domain_tensor in domains.items():
                    for logger in loggers:
                        self.log_samples(
                            logger,
                            pl_module,
                            domain_tensor,
                            domain_name,
                            f"ref_{'-'.join(domain_names)}_{domain_name}",
                            trainer,  # Pass trainer here
                        )


        latent_groups = pl_module.encode_domains(samples)
        for domain_group, domains in latent_groups.items():
            for domain, tensor in domains.items():
                if domain == "image_latents":
                    pl_module.domain_mods["image_latents"].vae_model.eval()
                    latent_groups[domain_group][domain] = pl_module.domain_mods["image_latents"].vae_model.encode(tensor)[0].flatten(start_dim=1)

                    self.log_samples(loggers[0] if loggers else None, pl_module, pl_module.domain_mods["image_latents"].vae_model(tensor)[0], domain, "forward_by_hand", trainer)
        selection_mod = SingleDomainSelection()

        with torch.no_grad():
            pl_module.eval()
            prediction_demi_cycles = batch_demi_cycles(
                pl_module.gw_mod, selection_mod, latent_groups
            ) 
            prediction_cycles = batch_cycles(
                pl_module.gw_mod,
                selection_mod,
                latent_groups,
                pl_module.domain_mods.keys(),
            )
            prediction_translations = batch_translations(
                pl_module.gw_mod, selection_mod, latent_groups
            )
        for logger in loggers:
            pl_module.eval()

            for domain_s, prediction in prediction_demi_cycles.items():
                self.log_samples(
                    logger,
                    pl_module,
                    pl_module.decode_domain(prediction, domain_s),
                    domain_s,
                    f"pred_dcy_{domain_s}",
                    trainer,
                )
            for (domain_s, domain_t), prediction in prediction_cycles.items():
                self.log_samples(
                    logger,
                    pl_module,
                    pl_module.decode_domain(prediction, domain_s),
                    domain_s,
                    f"pred_cy_{domain_s}_in_{domain_t}",
                    trainer,
                )
            for (
                domain_s,
                domain_t,
            ), prediction in prediction_translations.items():
                self.log_samples(
                    logger,
                    pl_module,
                    pl_module.decode_domain(prediction, domain_t),
                    domain_t,
                    f"pred_trans_{domain_s}_to_{domain_t}",
                    trainer,
                )

        pl_module.train()

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode != "train":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(trainer.current_epoch, trainer.loggers, pl_module, trainer)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode != "val":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(trainer.current_epoch, trainer.loggers, pl_module, trainer)

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode != "test":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        return self.on_callback(trainer.current_epoch, trainer.loggers, pl_module, trainer)

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode == "test":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        return self.on_callback(trainer.current_epoch, trainer.loggers, pl_module, trainer)
    
    def log_samples(
        self,
        logger: Logger,
        pl_module: GlobalWorkspaceBase,
        samples: Any,
        domain: str,
        mode: str,
        trainer: pl.Trainer,
    ) -> None:
        if not isinstance(logger, WandbLogger):
            print("Only logging to wandb is supported")
            print("The script will now log images locally in a very"
                " undesirable fashion. Beware.")
            self.log_locally(samples, mode)
            return

        match domain:
            case "image_latents":
                assert "image_latents" in pl_module.domain_mods
                self.log_visual_samples(logger, samples, mode)
            case _:
                print(f"Unsupported domain: {domain}")


    def log_visual_samples(
        self,
        logger: WandbLogger,
        samples: Any,
        mode: str,
    ) -> None:
        samples = np.clip(samples.cpu(),0.,1.)
        images = make_grid(samples, nrow=self.ncols, pad_value=1, normalize=False)
        logger.log_image(key=f"{self.log_key}/{mode}", images=[images])


    def log_locally(self, samples: Any, mode: str) -> None:
        print("called log locally !")
        # Create grid of images
        print("range: ", samples.max(), samples.min())
        samples = np.clip(samples.cpu(),0.,1.)
        images = make_grid(samples, nrow=self.ncols, pad_value=1, normalize=False)
        print("range: ", images.max(), images.min())
        
        # Define the directory to save images
        save_dir = "images"
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Find the next available filename
        base_path = os.path.join(save_dir, f"image_{mode}")
        image_path = base_path + ".png"
        index = 0
        
        while os.path.exists(image_path):
            index += 1
            image_path = f"{base_path}_{index}.png"
        
        # Save the image locally
        save_image(images, image_path)

            