# Shimmer
_GLoW, but very light_

This repository is the barebones API to build a global workspace.
It contains:
- the GlobalWorkspace module (using LightningModule) in two flavors: deterministic or variational;
- an interface for your expert domains that are connected to the GlobalWorkspace;
- a generic and barebone VAE class;
- a helper function to load configuration from yaml using OmegaConf.


## Install
Use it as a requirement in your own project.

For now, there is no PyPI entry, so in your requirements.txt:
```
shimmer@git+github.com:bdvllrs/shimmer.git@0.1.0#egg=shimmer
```

or in your pyproject.toml (with poetry for example):
```
shimmer = {git = "git@github.com:bdvllrs/shimmer.git", rev = "0.1.0"}
```

## Make a domain

```python
import torch

from shimmer.modules.domain import DomainModule


# DomainModule is a LightningModule, so you can 
class MyDomain(DomainModule):
    def encode(self, x: Any) -> torch.Tensor:
        # encode the input x into a latent representation
        # provided to the GW
        ...

    def decode(self, z: torch.Tensor) -> Any:
        # decode the latent representation back into the input form
        ...

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # must at least contain a key "loss" with the domain loss.
        losses = super().compute_loss(pred, target)
        # super computes an MSE.
        ...
        return losses

```

The GlobalWorkspace expects a DomainDescription dataclass.
It contains the DomainModule, and additional information, such as the latent dim of the
module, and how to configure the encoders and decoders.
