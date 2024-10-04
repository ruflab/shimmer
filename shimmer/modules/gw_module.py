from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TypedDict

import torch
from torch import nn

from shimmer.modules.domain import DomainModule
from shimmer.modules.selection import SelectionBase
from shimmer.types import (
    LatentsDomainGroupDT,
    LatentsDomainGroupT,
)


def translation(
    gw_module: "GWModuleBase",
    selection_mod: SelectionBase,
    x: LatentsDomainGroupT,
    to: str,
) -> torch.Tensor:
    """
    Translate from multiple domains to one domain.

    Args:
        gw_module (`"GWModuleBase"`): GWModule to perform the translation over
        selection_mod (`SelectionBase`): selection module
        x (`LatentsDomainGroupT`): the group of latent representations
        to (`str`): the domain name to encode to

    Returns:
        `torch.Tensor`: the translated unimodal representation
            of the provided domain.
    """
    return gw_module.decode(gw_module.encode_and_fuse(x, selection_mod), domains={to})[
        to
    ]


def cycle(
    gw_module: "GWModuleBase",
    selection_mod: SelectionBase,
    x: LatentsDomainGroupT,
    through: str,
) -> LatentsDomainGroupDT:
    """
    Do a full cycle from a group of representation through one domain.

    [Original domains] -> [GW] -> [through] -> [GW] -> [Original domains]

    Args:
        gw_module (`"GWModuleBase"`): GWModule to perform the translation over
        selection_mod (`SelectionBase`): selection module
        x (`LatentsDomainGroupT`): group of unimodal latent representation
        through (`str`): domain name to cycle through
    Returns:
        `LatentsDomainGroupDT`: group of unimodal latent representation after
            cycling.
    """
    return {
        domain: translation(
            gw_module,
            selection_mod,
            {through: translation(gw_module, selection_mod, x, through)},
            domain,
        )
        for domain in x
    }


def broadcast(
    gw_mod: "GWModuleBase",
    selection_mod: SelectionBase,
    latents: LatentsDomainGroupT,
) -> dict[str, torch.Tensor]:
    """
    broadcast a group

    Args:
        gw_mod (`"GWModuleBase"`): GWModule to perform the translation over
        selection_mod (`SelectionBase`): selection module
        latents (`LatentsDomainGroupT`): the group of latent representations

    Returns:
        `torch.Tensor`: the broadcast representation
    """
    predictions: dict[str, torch.Tensor] = {}
    state = gw_mod.encode_and_fuse(latents, selection_mod)
    all_domains = list(gw_mod.domain_mods.keys())
    for domain in all_domains:
        predictions[domain] = gw_mod.decode(state, domains=[domain])[domain]
    return predictions


def broadcast_cycles(
    gw_mod: "GWModuleBase",
    selection_mod: SelectionBase,
    latents: LatentsDomainGroupT,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    broadcast a group

    Args:
        gw_mod (`"GWModuleBase"`): GWModule to perform the translation over
        selection_mod (`SelectionBase`): selection module
        latents (`LatentsDomainGroupT`): the group of latent representations

    Returns:
        `torch.Tensor`: the broadcast representation
    """
    all_domains = list(latents.keys())
    predictions = broadcast(gw_mod, selection_mod, latents)
    inverse = {
        name: latent for name, latent in predictions.items() if name not in all_domains
    }
    cycles: dict[str, torch.Tensor] = {}
    if len(inverse):
        cycles = broadcast(gw_mod, selection_mod, inverse)
    return predictions, cycles


def get_n_layers(n_layers: int, hidden_dim: int) -> list[nn.Module]:
    """
    Makes a list of `n_layers` `nn.Linear` layers with `nn.ReLU`.

    Args:
        n_layers (`int`): number of layers
        hidden_dim (`int`): size of the hidden dimension

    Returns:
        `list[nn.Module]`: list of linear and relu layers.
    """
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return layers


class GWDecoder(nn.Sequential):
    """A Decoder network for GWModules."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        """
        Initializes the decoder.

        Args:
            in_dim (`int`): input dimension
            hidden_dim (`int`): hidden dimension
            out_dim (`int`): output dimension
            n_layers (`int`): number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after).
        """

        self.in_dim = in_dim
        """input dimension"""

        self.hidden_dim = hidden_dim
        """hidden dimension"""

        self.out_dim = out_dim
        """output dimension"""

        self.n_layers = n_layers
        """
        number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after)."""

        super().__init__(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )


class GWEncoder(GWDecoder):
    """
    An Encoder network used in GWModules.

    This is similar to the decoder, but adds a tanh non-linearity at the end.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        """
        Initializes the encoder.

        Args:
            in_dim (`int`): input dimension
            hidden_dim (`int`): hidden dimension
            out_dim (`int`): output dimension
            n_layers (`int`): number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after).
        """
        super().__init__(in_dim, hidden_dim, out_dim, n_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


class GWModulePrediction(TypedDict):
    """TypedDict of the output given when calling `GlobalWorkspaceBase.predict`"""

    states: torch.Tensor
    """
    GW state representation from domain groups with only one domain.
    The key represent the domain's name.
    """

    broadcasts: dict[str, torch.Tensor]
    """
    broadcasts predictions of the model for each domain. It contains demi-cycles,
    translations, and fused.
    """

    cycles: dict[str, torch.Tensor]
    """
    Cycle predictions of the model from one domain through another one.
    """


class GWModuleBase(nn.Module, ABC):
    """
    Base class for GWModule.

    GWModule handles encoding, decoding the unimodal representations
    using the `gw_encoders` and`gw_decoders`, and define
    some common operations in GW like cycles and translations.

    This is an abstract class and should be implemented.
    For an implemented interface, see `GWModule`.
    """

    def __init__(
        self,
        domain_mods: Mapping[str, DomainModule],
        workspace_dim: int,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the GWModule.

        Args:
            domain_modules (`Mapping[str, DomainModule]`): the domain modules.
            workspace_dim (`int`): dimension of the GW.
        """
        super().__init__()

        self.domain_mods = domain_mods
        """The unimodal domain modules."""

        self.workspace_dim = workspace_dim
        """Dimension of the GW"""

    @abstractmethod
    def fuse(
        self, x: LatentsDomainGroupT, selection_scores: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Merge function used to combine domains.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        ...

    @abstractmethod
    def encode(self, x: LatentsDomainGroupT) -> LatentsDomainGroupDT:
        """
        Encode the latent representation infos to the pre-fusion GW representation.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations

        Returns:
            `LatentsDomainGroupT`: pre-fusion GW representations
        """
        ...

    def encode_and_fuse(
        self, x: LatentsDomainGroupT, selection_module: SelectionBase
    ) -> torch.Tensor:
        """
        Encode the latent representation infos to the final GW representation.
        It combines the encode and fuse methods.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.

        Returns:
            `torch.Tensor`: The merged representation.
        """
        encodings = self.encode(x)
        selection_scores = selection_module(x, encodings)
        return self.fuse(encodings, selection_scores)

    @abstractmethod
    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> LatentsDomainGroupDT:
        """
        Decode the GW representation into given `domains`.

        Args:
            z (`torch.Tensor`): the GW representation.
            domains (`Iterable[str]`): iterable of domains to decode.

        Returns:
            `LatentsDomainGroupDT`: the decoded unimodal representations.
        """
        ...

    def forward(
        self,
        latent_domains: LatentsDomainGroupT,
        selection_module: SelectionBase,
    ) -> GWModulePrediction:
        """
        Computes demi-cycles, cycles, and translations.

        Args:
            latent_domains (`LatentsDomainGroupT`): Group of domains
            selection_module (`SelectionBase`): selection module

        Returns:
            `GWModulePredictions`: the predictions on the group.
        """
        broadcasts, cycles = broadcast_cycles(self, selection_module, latent_domains)

        return GWModulePrediction(
            states=self.encode_and_fuse(latent_domains, selection_module),
            broadcasts=broadcasts,
            cycles=cycles,
        )


class GWModule(GWModuleBase):
    """GW nn.Module. Implements `GWModuleBase`."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
    ) -> None:
        """
        Initializes the GWModule.

        Args:
            domain_modules (`Mapping[str, DomainModule]`): the domain modules.
            workspace_dim (`int`): dimension of the GW.
            gw_encoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a an torch.nn.Module class that encodes a
                unimodal latent representations into a GW representation (pre fusion).
            gw_decoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a an torch.nn.Module class that decodes a
                 GW representation to a unimodal latent representation.
        """
        super().__init__(domain_modules, workspace_dim)

        self.gw_encoders = nn.ModuleDict(gw_encoders)
        """The module's encoders"""

        self.gw_decoders = nn.ModuleDict(gw_decoders)
        """The module's decoders"""

    def fuse(
        self,
        x: LatentsDomainGroupT,
        selection_scores: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Merge function used to combine domains.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        return torch.tanh(
            torch.sum(
                torch.stack(
                    [
                        selection_scores[domain].unsqueeze(1) * x[domain]
                        for domain in selection_scores
                    ]
                ),
                dim=0,
            )
        )

    def encode(self, x: LatentsDomainGroupT) -> LatentsDomainGroupDT:
        """
        Encode the latent representation infos to the pre-fusion GW representation.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations.

        Returns:
            `LatentsDomainGroupT`: pre-fusion representation
        """
        return {
            domain_name: self.gw_encoders[domain_name](domain)
            for domain_name, domain in x.items()
        }

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> LatentsDomainGroupDT:
        """
        Decodes a GW representation to multiple domains.

        Args:
            z (`torch.Tensor`): the GW representation
            domains (`Iterable[str] | None`): the domains to decode to. Defaults to
                use keys in `gw_interfaces` (all domains).
        Returns:
            `LatentsDomainGroupDT`: decoded unimodal representation
        """
        return {
            domain: self.gw_decoders[domain](z)
            for domain in domains or self.gw_decoders.keys()
        }
