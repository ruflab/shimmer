# Shimmer
_GLoW, but very light_

This repository is the barebones API to build a global workspace.
It contains:
- the GlobalWorkspace module (using LightningModule) in two flavors: deterministic or variational;
- an interface for your expert domains that are connected to the GlobalWorkspace;
- a generic and barebone VAE class;


## Install
Use it as a requirement in your own project.

For now, there is no PyPI entry, so in your requirements.txt:
```
shimmer@git+github.com:bdvllrs/shimmer.git@main#egg=shimmer
```

or in your pyproject.toml (with poetry for example):
```
shimmer = {git = "git@github.com:bdvllrs/shimmer.git", rev = "main"}
```

## Make a domain

```python
from typing import Any

import torch
import torch.nn.functional as F
from shimmer import DomainModule, LossOutput


class MyDomain(DomainModule):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)

    def encode(self, x: Any) -> torch.Tensor:
        # encode the input x into a latent representation
        # provided to the GW
        ...

    def decode(self, z: torch.Tensor) -> Any:
        # decode the latent representation back into the input form
        ...

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        return LossOutput(
            loss=F.mse_loss(pred, target),  # loss used for training
            metrics={},  # additional metrics to log (not required)
        )
```

## Use a GW

### GWInterface
To link each domain module with the global workspace, we need a GWInterface.
It encodes the unimodal representations into a GW representation, or decodes a
GW representation into a unimodal representation.

```python
from shimmer import GWInterface

my_domain = MyDomain(latent_dim=32)
my_domain_gw_interface = GWInterface(
    my_domain,
    workspace_dim=12,  # latent dim of the global workspace
    encoder_hidden_dim=32,  # hidden dimension for the GW encoder
    encoder_n_layers=3,  # n layers to use for the GW encoder
    decoder_hidden_dim=32,  # hidden dimension for the GW decoder
    decoder_n_layers=3,  # n layers to use for the GW decoder
)
```


### GW
To load a global workspace, use: 
```python
from shimmer import GlobalWorkspace, LossCoefs

domain_modules = {"my_domain": my_domain}

gw_interfaces = {"my_domain": my_domain_gw_interface}

workspace_dim = 32

loss_coefs: LossCoefs = {
    "translations": 1.0,
    "demi_cycles": 0.0,
    "cycles": 1.0,
    "contrastives": 0.1,
}

model = GlobalWorkspace(
    domain_modules,
    gw_interfaces,
    workspace_dim,
    loss_coefs,
)
```


# Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).
