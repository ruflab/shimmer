import torch.utils.data
from utils import DummyData, DummyDataset, DummyDomainModule

from shimmer import GlobalWorkspace, GWDecoder, GWEncoder


def test_training():
    train_dataset = DummyDataset(
        size=128,
        domains=["v", "t"],
    )

    domains = {
        "v": DummyDomainModule(latent_dim=128),
        "t": DummyDomainModule(latent_dim=128),
        "a": DummyDomainModule(latent_dim=128),
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
        "a": GWEncoder(
            domains["a"].latent_dim,
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
        "a": GWDecoder(
            workspace_dim,
            hidden_dim=64,
            out_dim=domains["a"].latent_dim,
            n_layers=1,
        ),
    }

    gw = GlobalWorkspace(
        domains,
        gw_encoders,
        gw_decoders,
        workspace_dim=16,
        loss_coefs={},
    )

    batch_size = 32
    train_datalader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    batch = next(iter(train_datalader))

    assert batch.keys() == {"v", "t"}
    assert isinstance(batch["v"], DummyData)

    unimodal_latents: dict[str, torch.Tensor] = {
        domain: domains[domain].encode(batch[domain]) for domain in batch
    }
    assert isinstance(unimodal_latents["v"], torch.Tensor)
    assert unimodal_latents["v"].size() == (32, 128)

    selection_scores = {
        domain: torch.full((batch_size,), 1.0 / len(unimodal_latents))
        for domain in unimodal_latents.keys()
    }
    workspace_latent = gw.encode_and_fuse(unimodal_latents, selection_scores)

    assert workspace_latent.size() == (32, 16)

    reconstructed_unimodal_latents = gw.decode(workspace_latent, domains={"v", "a"})

    assert reconstructed_unimodal_latents.keys() == {"v", "a"}
    assert reconstructed_unimodal_latents["v"].size() == (32, 128)
