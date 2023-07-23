import torch.utils.data
from utils import DummyData, DummyDataset, DummyDomainModule

from shimmer.modules.domain import DomainDescription
from shimmer.modules.global_workspace import DeterministicGlobalWorkspace


def test_training():
    train_dataset = DummyDataset(
        size=128,
        domains=["v", "t"],
    )

    domains = {
        "v": DummyDomainModule(),
        "t": DummyDomainModule(),
    }
    domain_description = {
        "v": DomainDescription(
            module=domains["v"],
            latent_dim=128,
            encoder_hidden_dim=64,
            encoder_n_layers=1,
            decoder_hidden_dim=64,
            decoder_n_layers=1,
        ),
        "t": DomainDescription(
            module=domains["t"],
            latent_dim=128,
            encoder_hidden_dim=64,
            encoder_n_layers=1,
            decoder_hidden_dim=64,
            decoder_n_layers=1,
        ),
    }

    global_workspace = DeterministicGlobalWorkspace(
        domain_description,
        latent_dim=16,
        loss_coefficients={},
    )

    train_datalader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

    batch = next(iter(train_datalader))

    assert batch.keys() == {"v", "t"}
    assert isinstance(batch["v"], DummyData)

    unimodal_latents = {
        domain: domains[domain].encode(batch[domain]) for domain in domains
    }
    assert isinstance(unimodal_latents["v"], torch.Tensor)
    assert unimodal_latents["v"].size() == (32, 128)

    workspace_latent = global_workspace.encode(unimodal_latents)

    assert workspace_latent.size() == (32, 16)

    reconstructed_unimodal_latents = global_workspace.decode(
        workspace_latent, domains={"v", "a"}
    )

    assert reconstructed_unimodal_latents.keys() == {"v", "a"}
    assert reconstructed_unimodal_latents["v"].size() == (32, 128)
