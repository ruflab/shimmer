from pathlib import Path

import torch.utils.data
from utils import DummyDomainModule

from shimmer import GlobalWorkspace2Domains, GWDecoder, GWEncoder

here = Path(__file__).parent


def save_gw_2_domains():
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

    torch.save({"state_dict": gw.state_dict()}, here / "data" / "gw_2_domains.ckpt")


if __name__ == "__main__":
    save_gw_2_domains()
