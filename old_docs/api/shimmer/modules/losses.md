Module shimmer.modules.losses
=============================

Functions
---------

    
`sample_scaling_factors(binary_scaling_prob: float, batch_size: int, temperature: float, device: torch.device)`
:   Args:
        binary_scaling_prob: float
        batch_size: int
        temperature: float greater than 0

Classes
-------

`GWLosses(gw_mod: shimmer.modules.gw_module.GWModule, domain_mods: dict[str, shimmer.modules.domain.DomainModule], loss_coefs: shimmer.modules.losses.LossCoefs, contrastive_fn: collections.abc.Callable[[torch.Tensor, torch.Tensor], shimmer.modules.domain.LossOutput])`
:   Base Abstract Class for Global Workspace (GW) losses. This module is used
    to compute the different losses of the GW (typically translation, cycle,
    demi-cycle, contrastive losses).
    
    Main loss module to use with the GlobalWorkspace
    Args:
        gw_mod: the GWModule
        domain_mods: a dict where the key is the domain name and
            value is the DomainModule
        loss_coefs: loss coefficients. LossCoefs object, or a mapping to float with
            correct keys.
        contrastive_fn: the contrastive function to use in contrastive loss

    ### Ancestors (in MRO)

    * shimmer.modules.losses.GWLossesBase
    * torch.nn.modules.module.Module
    * abc.ABC

    ### Methods

    `contrastive_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `cycle_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `demi_cycle_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `translation_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

`GWLossesBase(*args, **kwargs)`
:   Base Abstract Class for Global Workspace (GW) losses. This module is used
    to compute the different losses of the GW (typically translation, cycle,
    demi-cycle, contrastive losses).
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module
    * abc.ABC

    ### Descendants

    * shimmer.modules.losses.GWLosses
    * shimmer.modules.losses.GWLossesFusion
    * shimmer.modules.losses.VariationalGWLosses

    ### Methods

    `step(self, domain_latents: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]], mode: str) ‑> shimmer.modules.domain.LossOutput`
    :   Computes the losses
        Args:
            domain_latents: All latent groups
            mode: train/val/test
        Returns: LossOutput object

`GWLossesFusion(gw_mod: shimmer.modules.gw_module.GWModule, domain_mods: dict[str, shimmer.modules.domain.DomainModule], contrastive_fn: collections.abc.Callable[[torch.Tensor, torch.Tensor], shimmer.modules.domain.LossOutput])`
:   Base Abstract Class for Global Workspace (GW) losses. This module is used
    to compute the different losses of the GW (typically translation, cycle,
    demi-cycle, contrastive losses).
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * shimmer.modules.losses.GWLossesBase
    * torch.nn.modules.module.Module
    * abc.ABC

    ### Methods

    `broadcast_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]], mode: str) ‑> dict[str, torch.Tensor]`
    :

    `contrastive_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `cycle_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `demi_cycle_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `translation_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

`LossCoefs(*args, **kwargs)`
:   Dict of loss coefficients used in the GlobalWorkspace
    If one is not provided, the coefficient is assumed to be 0 and will not be logged.
    If the loss is excplicitely set to 0, it will be logged, but not take part in
    the total loss.

    ### Ancestors (in MRO)

    * builtins.dict

    ### Class variables

    `contrastives: float`
    :

    `cycles: float`
    :

    `demi_cycles: float`
    :

    `translations: float`
    :

`VariationalGWLosses(gw_mod: shimmer.modules.gw_module.VariationalGWModule, domain_mods: dict[str, shimmer.modules.domain.DomainModule], loss_coefs: shimmer.modules.losses.VariationalLossCoefs, contrastive_fn: collections.abc.Callable[[torch.Tensor, torch.Tensor], shimmer.modules.domain.LossOutput] | None = None, var_contrastive_fn: collections.abc.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], shimmer.modules.domain.LossOutput] | None = None)`
:   Base Abstract Class for Global Workspace (GW) losses. This module is used
    to compute the different losses of the GW (typically translation, cycle,
    demi-cycle, contrastive losses).
    
    Variational loss module to use with the VariationalGlobalWorkspace
    Args:
        gw_mod: the GWModule
        domain_mods: a dict where the key is the domain name and
            value is the DomainModule
        loss_coefs: loss coefficients. LossCoefs object, or a mapping to float with
            correct keys.
        contrastive_fn: the contrastive function to use in contrastive loss
        var_contrastive_fn: a contrastive function that uses uncertainty

    ### Ancestors (in MRO)

    * shimmer.modules.losses.GWLossesBase
    * torch.nn.modules.module.Module
    * abc.ABC

    ### Methods

    `contrastive_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `cycle_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `demi_cycle_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `kl_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `translation_loss(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

`VariationalLossCoefs(*args, **kwargs)`
:   dict() -> new empty dictionary
    dict(mapping) -> new dictionary initialized from a mapping object's
        (key, value) pairs
    dict(iterable) -> new dictionary initialized as if via:
        d = {}
        for k, v in iterable:
            d[k] = v
    dict(**kwargs) -> new dictionary initialized with the name=value pairs
        in the keyword argument list.  For example:  dict(one=1, two=2)

    ### Ancestors (in MRO)

    * builtins.dict

    ### Class variables

    `contrastives: float`
    :

    `cycles: float`
    :

    `demi_cycles: float`
    :

    `kl: float`
    :

    `translations: float`
    :