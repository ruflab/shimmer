from utils import DummyDomainModuleWithParams

from shimmer import GlobalWorkspace2Domains, GWDecoder, GWEncoder


def test_training():
    domains = {
        "v": DummyDomainModuleWithParams(latent_dim=128),
        "t": DummyDomainModuleWithParams(latent_dim=128),
        "a": DummyDomainModuleWithParams(latent_dim=128),
    }

    domains["a"].unfreeze()

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

    gw = GlobalWorkspace2Domains(
        domains,
        gw_encoders,
        gw_decoders,
        workspace_dim=16,
        loss_coefs={},
    )
    assert gw.domain_mods["v"].is_frozen
    assert not gw.domain_mods["a"].is_frozen
    assert not len([p for p in gw.domain_mods["v"].parameters() if p.requires_grad])
    assert len([p for p in gw.domain_mods["a"].parameters() if p.requires_grad])
