import torch.utils.data
from utils import DummyData, DummyDataset, DummyDomainModule

from shimmer import GlobalWorkspace, GWInterface


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

    gw_latent_dim = 16

    gw_interfaces = {
        "v": GWInterface(
            domains["v"],
            gw_latent_dim=gw_latent_dim,
            encoder_hidden_dim=64,
            encoder_n_layers=1,
            decoder_hidden_dim=64,
            decoder_n_layers=1,
        ),
        "t": GWInterface(
            domains["t"],
            gw_latent_dim=gw_latent_dim,
            encoder_hidden_dim=64,
            encoder_n_layers=1,
            decoder_hidden_dim=64,
            decoder_n_layers=1,
        ),
        "a": GWInterface(
            domains["a"],
            gw_latent_dim=gw_latent_dim,
            encoder_hidden_dim=64,
            encoder_n_layers=1,
            decoder_hidden_dim=64,
            decoder_n_layers=1,
        ),
    }

    gw = GlobalWorkspace(
        domains,
        gw_interfaces,
        gw_latent_dim=16,
        loss_coefs={},
    )

    train_datalader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

    batch = next(iter(train_datalader))

    assert batch.keys() == {"v", "t"}
    assert isinstance(batch["v"], DummyData)

    unimodal_latents = {
        domain: domains[domain].encode(batch[domain]) for domain in batch
    }
    assert isinstance(unimodal_latents["v"], torch.Tensor)
    assert unimodal_latents["v"].size() == (32, 128)

    workspace_latent = gw.encode(unimodal_latents)

    assert workspace_latent.size() == (32, 16)

    reconstructed_unimodal_latents = gw.decode(
        workspace_latent, domains={"v", "a"}
    )

    assert reconstructed_unimodal_latents.keys() == {"v", "a"}
    assert reconstructed_unimodal_latents["v"].size() == (32, 128)
