import torch.utils.data
from utils import DummyData, DummyDataset, DummyDomainModule

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

    global_workspace = DeterministicGlobalWorkspace(
        domains={"v", "t", "a"},
        latent_dim=16,
        input_dim={"v": 128, "t": 128, "a": 128},
        encoder_hidden_dim={"v": 64, "t": 64, "a": 64},
        encoder_n_layers={"v": 1, "t": 1, "a": 1},
        decoder_hidden_dim={"v": 64, "t": 64, "a": 64},
        decoder_n_layers={"v": 1, "t": 1, "a": 1},
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
