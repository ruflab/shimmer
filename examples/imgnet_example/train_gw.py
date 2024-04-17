from dataset import prepare_datasets
from domains import ImageDomain, TextDomain
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from shimmer import GlobalWorkspace, GWDecoder, GWEncoder, LossCoefs
from shimmer.modules.global_workspace import SchedulerArgs


def train_gw():
    # Prepare data modules from the dataset script
    train_data_module, val_data_module = prepare_datasets()

    # Initialize the domain modules with the specific
    # latent dimensions and model parameters
    image_domain = ImageDomain(latent_dim=384)
    text_domain = TextDomain(latent_dim=384)

    domain_mods = {
        "image_latents": image_domain,
        "caption_embeddings": text_domain,
    }

    workspace_dim = 16  # Define the dimension of the global workspace

    # Define modality encoders and decoders
    gw_encoders = {}
    gw_decoders = {}
    for name, mod in domain_mods.items():
        gw_encoders[name] = GWEncoder(
            mod.latent_dim,
            hidden_dim=64,
            out_dim=workspace_dim,
            n_layers=1,
        )
        gw_decoders[name] = GWDecoder(
            in_dim=workspace_dim,
            hidden_dim=64,
            out_dim=mod.latent_dim,
            n_layers=1,
        )

    # Loss coefficients setup
    loss_coefs: LossCoefs = {
        "translations": 1.0,
        "demi_cycles": 1.0,
        "cycles": 1.0,
        "contrastives": 0.01,
    }

    n_epochs = 4  # Number of training epochs

    global_workspace = GlobalWorkspace(
        domain_mods,
        gw_encoders,
        gw_decoders,
        workspace_dim,
        loss_coefs,
        scheduler_args=SchedulerArgs(
            max_lr=1e-3,
            total_steps=n_epochs
            * len(
                train_data_module.train_datasets["image_latents", "caption_embeddings"]
            )
            // 64,
        ),
    )

    # Trainer setup
    trainer = Trainer(
        devices=1,  # assuming training on 1 GPU
        max_epochs=n_epochs,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="global_workspace_{epoch}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
            ),
        ],
    )

    # Run training and validation
    trainer.fit(global_workspace, train_data_module, val_data_module)
    trainer.validate(global_workspace, val_data_module, "best")


if __name__ == "__main__":
    train_gw()
