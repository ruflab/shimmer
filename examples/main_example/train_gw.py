from dataset import GWDataModule, domain_sizes, get_domain_data, make_datasets
from domains import GenericDomain
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from shimmer import GlobalWorkspace, GWDecoder, GWEncoder, LossCoefs
from shimmer.modules.global_workspace import SchedulerArgs


def train_gw():
    train_domain1, val_domain1 = get_domain_data("domain1")
    train_domain2, val_domain2 = get_domain_data("domain2")

    train_data = {"domain1": train_domain1, "domain2": train_domain2}
    val_data = {"domain1": val_domain1, "domain2": val_domain2}

    n_paired = 32

    train_datasets = make_datasets(
        train_data, paired_items=list(range(n_paired)), add_unpaired_dataset=True
    )

    # The val set is completely paired and we do not add the unpaired datasets
    val_datasets = make_datasets(val_data, list(range(val_domain1.size(0))))

    batch_size = 32

    data = GWDataModule(val_datasets, train_datasets, batch_size=batch_size)

    # We have pretrained the domain module, we need to load them.
    domain_mod1 = GenericDomain.load_from_checkpoint(
        "checkpoints/domain1.ckpt",
        input_size=domain_sizes["domain1"],
        latent_dim=32,
    )
    domain_mod2 = GenericDomain.load_from_checkpoint(
        "checkpoints/domain2.ckpt",
        input_size=domain_sizes["domain2"],
        latent_dim=32,
    )

    domain_mods = {
        "domain1": domain_mod1,
        "domain2": domain_mod2,
    }

    workspace_dim = 16

    # Now we define modality encoders and decoders that will encode and decode
    # the domain representations to and from the global workspace
    gw_encoders: dict[str, nn.Module] = {}
    gw_decoders: dict[str, nn.Module] = {}
    for name, mod in domain_mods.items():
        gw_encoders[name] = GWEncoder(
            mod.latent_dim,
            hidden_dim=64,
            out_dim=workspace_dim,
            # total number of Linear layers is this value + 2 (one before, one after)
            n_layers=1,
        )
        gw_decoders[name] = GWDecoder(
            in_dim=workspace_dim,
            hidden_dim=64,
            out_dim=mod.latent_dim,
            # total number of Linear layers is this value + 2 (one before, one after)
            n_layers=1,
        )

    loss_coefs: LossCoefs = {
        "translations": 1.0,
        "demi_cycles": 1.0,
        "cycles": 1.0,
        "contrastives": 0.01,
    }

    n_epochs = 4

    global_workspace = GlobalWorkspace(
        domain_mods,
        gw_encoders,
        gw_decoders,
        workspace_dim,
        loss_coefs,
        scheduler_args=SchedulerArgs(
            max_lr=1e-3, total_steps=n_epochs * train_domain1.size(0) // batch_size
        ),
    )

    trainer = Trainer(
        devices=1,
        max_epochs=n_epochs,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="gw",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
            ),
        ],
    )
    trainer.fit(global_workspace, data)
    trainer.validate(global_workspace, data, "best")


if __name__ == "__main__":
    train_gw()
