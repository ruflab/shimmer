from pathlib import Path

import torch
import torch.utils.data
from migrate_ckpt import migrate_from_folder
from utils import DummyDomainModule

from shimmer import GlobalWorkspace2Domains, GWDecoder, GWEncoder
from shimmer.utils import MIGRATION_DIR

here = Path(__file__).parent


def test_ckpt_migration_2_domains():
    old_ckpt_path = here / "data" / "old_gw_2_domains.ckpt"
    old_ckpt = torch.load(old_ckpt_path, weights_only=True)
    new_ckpt, done_migrations = migrate_from_folder(old_ckpt, MIGRATION_DIR)

    old_keys = set(old_ckpt["state_dict"].keys())
    new_keys = set(new_ckpt["state_dict"].keys())
    print(f"Removed keys: {old_keys - new_keys}")
    print(f"New keys: {new_keys - old_keys}")

    print("Done migrations:", ", ".join(map(lambda x: x.name, done_migrations)))

    domains = {
        "v": DummyDomainModule(latent_dim=32),
        "t": DummyDomainModule(latent_dim=32),
    }

    workspace_dim = 16

    gw_encoders = {
        "v": GWEncoder(
            domains["v"].latent_dim,
            hidden_dim=64,
            out_dim=workspace_dim,
            n_layers=1,
        ),
        "t": GWEncoder(
            domains["t"].latent_dim,
            hidden_dim=64,
            out_dim=workspace_dim,
            n_layers=1,
        ),
    }

    gw_decoders = {
        "v": GWDecoder(
            workspace_dim,
            hidden_dim=64,
            out_dim=domains["v"].latent_dim,
            n_layers=1,
        ),
        "t": GWDecoder(
            workspace_dim,
            hidden_dim=64,
            out_dim=domains["t"].latent_dim,
            n_layers=1,
        ),
    }

    gw = GlobalWorkspace2Domains(
        domains,
        gw_encoders,
        gw_decoders,
        workspace_dim=16,
        loss_coefs={},
    )

    gw.load_state_dict(new_ckpt["state_dict"])
