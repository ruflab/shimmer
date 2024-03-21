from collections.abc import Iterable

import torch

from shimmer.modules.gw_module import GWModuleBase, GWModuleWithUncertainty
from shimmer.types import (
    LatentsDomainGroupDT,
    LatentsDomainGroupsT,
    LatentsDomainGroupT,
)


def translation(
    gw_module: GWModuleBase, x: LatentsDomainGroupT, to: str
) -> torch.Tensor:
    """Translate from multiple domains to one domain.

    Args:
        gw_module (`GWModuleBase`): GWModule to perform the translation over
        x (`LatentsDomainGroupT`): the group of latent representations
        to (`str`): the domain name to encode to

    Returns:
        `torch.Tensor`: the translated unimodal representation
            of the provided domain.
    """
    return gw_module.decode(gw_module.encode(x), domains={to})[to]


def translation_uncertainty(
    gw_module: GWModuleWithUncertainty, x: LatentsDomainGroupT, to: str
) -> torch.Tensor:
    """Translate a latent representation to a specified domain.

    Args:
        x (`LatentsDomainGroupT`): group of latent representations.
        to (`str`): domain name to translate to.

    Returns:
        `torch.Tensor`: translated unimodal representation in domain given in `to`.
    """
    return gw_module.decode(gw_module.encoded_mean(x), domains={to})[to]


def cycle(
    gw_module: GWModuleBase, x: LatentsDomainGroupT, through: str
) -> LatentsDomainGroupDT:
    """Do a full cycle from a group of representation through one domain.

    [Original domains] -> [GW] -> [through] -> [GW] -> [Original domains]

    Args:
        gw_module (`GWModuleBase`): GWModule to perform the translation over
        x (`LatentsDomainGroupT`): group of unimodal latent representation
        through (`str`): domain name to cycle through
    Returns:
        `LatentsDomainGroupDT`: group of unimodal latent representation after
            cycling.
    """
    return {
        domain: translation(
            gw_module, {through: translation(gw_module, x, through)}, domain
        )
        for domain in x.keys()
    }


def cycle_uncertainty(
    gw_module: GWModuleWithUncertainty, x: LatentsDomainGroupT, through: str
) -> LatentsDomainGroupDT:
    """Do a full cycle from a group of representation through one domain.

    [Original domains] -> [GW] -> [through] -> [GW] -> [Original domains]

    Args:
        x (`LatentsDomainGroupT`): group of unimodal latent representation
        through (`str`): domain name to cycle through
    Returns:
        `LatentsDomainGroupDT`: group of unimodal latent representation after
            cycling.
    """
    return {
        domain: translation_uncertainty(
            gw_module, {through: translation_uncertainty(gw_module, x, through)}, domain
        )
        for domain in x.keys()
    }


def batch_demi_cycles(
    gw_mod: GWModuleBase, latent_domains: LatentsDomainGroupsT
) -> dict[str, torch.Tensor]:
    """Computes demi-cycles of a batch of groups of domains.

    Args:
        gw_mod (`GWModuleBase`): the GWModuleBase
        latent_domains (`LatentsT`): the batch of groups of domains

    Returns:
        `dict[str, torch.Tensor]`: demi-cycles predictions for each domain.
    """
    predictions: dict[str, torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) > 1:
            continue
        domain_name = list(domains)[0]
        z = translation(gw_mod, latents, to=domain_name)
        predictions[domain_name] = z
    return predictions


def batch_cycles(
    gw_mod: GWModuleBase,
    latent_domains: LatentsDomainGroupsT,
    through_domains: Iterable[str],
) -> dict[tuple[str, str], torch.Tensor]:
    """Computes cycles of a batch of groups of domains.

    Args:
        gw_mod (`GWModuleBase`): GWModule to use for the cycle
        latent_domains (`LatentsT`): the batch of groups of domains
        out_domains (`Iterable[str]`): iterable of domain names to do the cycle through.
            Each domain will be done separetely.

    Returns:
        `dict[tuple[str, str], torch.Tensor]`: cycles predictions for each
            couple of (start domain, intermediary domain).
    """
    predictions: dict[tuple[str, str], torch.Tensor] = {}
    for domains_source, latents_source in latent_domains.items():
        if len(domains_source) > 1:
            continue
        domain_name_source = next(iter(domains_source))
        for domain_name_through in through_domains:
            if domain_name_source == domain_name_through:
                continue
            z = cycle(gw_mod, latents_source, through=domain_name_through)
            domains = (domain_name_source, domain_name_through)
            predictions[domains] = z[domain_name_source]
    return predictions


def batch_translations(
    gw_mod: GWModuleBase, latent_domains: LatentsDomainGroupsT
) -> dict[tuple[str, str], torch.Tensor]:
    """Computes translations of a batch of groups of domains.

    Args:
        gw_mod (`GWModuleBase`): GWModule to do the translation
        latent_domains (`LatentsT`): the batch of groups of domains

    Returns:
        `dict[tuple[str, str], torch.Tensor]`: translation predictions for each
            couple of (start domain, target domain).
    """
    predictions: dict[tuple[str, str], torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) < 2:
            continue
        for domain_name_source in domains:
            for domain_name_target in domains:
                if domain_name_source == domain_name_target:
                    continue
                prediction = translation(
                    gw_mod,
                    {domain_name_source: latents[domain_name_source]},
                    to=domain_name_target,
                )
                predictions[(domain_name_source, domain_name_target)] = prediction
    return predictions
