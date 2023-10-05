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
shimmer@git+github.com:bdvllrs/shimmer.git@0.1.2#egg=shimmer
```

or in your pyproject.toml (with poetry for example):
```
shimmer = {git = "git@github.com:bdvllrs/shimmer.git", rev = "0.1.2"}
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

## Use a GW

### DomainDescription
The GlobalWorkspace expects a DomainDescription dataclass.
It contains the DomainModule, and additional information, such as the latent dim of the
module, and how to configure the encoders and decoders.

```python
from shimmer.modules.domain import DomainDescription


my_domain = MyDomain()
my_domain_descr = DomainDescription(
    module=my_domain,
    latent_dim=12,  # latent dim of the domain module
    encoder_hidden_dim=32,  # hidden dimension for the GW encoder
    encoder_n_layers=3,  # n layers to use for the GW encoder
    decoder_hidden_dim=32,  # hidden dimension for the GW decoder
    decoder_n_layers=3,  # n layers to use for the GW decoder
)
```


### GW
There are two kinds of GW in the lib: Deterministic, and Variational.
Deterministic is the one used in [this paper](https://arxiv.org/abs/2306.15711).
The Variational one is under investigation.
