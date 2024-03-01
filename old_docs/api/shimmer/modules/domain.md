Module shimmer.modules.domain
=============================

Classes
-------

`DomainModule(latent_dim: int)`
:   Base class for a DomainModule.
    We do not use ABC here because some modules could be without encore or decoder.
    
    Args:
        latent_dim: latent dimension of the unimodal module
        encoder_hidden_dim: number of hidden

    ### Ancestors (in MRO)

    * lightning.pytorch.core.module.LightningModule
    * lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    * lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    * lightning.pytorch.core.hooks.ModelHooks
    * lightning.pytorch.core.hooks.DataHooks
    * lightning.pytorch.core.hooks.CheckpointHooks
    * torch.nn.modules.module.Module

    ### Methods

    `compute_broadcast_loss(self, pred: torch.Tensor, target: torch.Tensor) ‑> shimmer.modules.domain.LossOutput`
    :   Computes the loss for a broadcast (fusion). Override if the translation loss is
        different that the generic loss.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.

    `compute_cy_loss(self, pred: torch.Tensor, target: torch.Tensor) ‑> shimmer.modules.domain.LossOutput`
    :   Computes the loss for a cycle. Override if the cycle loss is
        different that the generic loss.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.

    `compute_dcy_loss(self, pred: torch.Tensor, target: torch.Tensor) ‑> shimmer.modules.domain.LossOutput`
    :   Computes the loss for a demi-cycle. Override if the demi-cycle loss is
        different that the generic loss.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.

    `compute_loss(self, pred: torch.Tensor, target: torch.Tensor) ‑> shimmer.modules.domain.LossOutput`
    :   Computes the loss of the modality. If you implement compute_dcy_loss,
        compute_cy_loss and compute_tr_loss independently, no need to define this
        function.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.

    `compute_tr_loss(self, pred: torch.Tensor, target: torch.Tensor) ‑> shimmer.modules.domain.LossOutput`
    :   Computes the loss for a translation. Override if the translation loss is
        different that the generic loss.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.

    `decode(self, z: torch.Tensor) ‑> Any`
    :   Decode data back to the domain data.
        Args:
            z: unimodal representation of the domain.
        Returns:
            the original domain data.

    `encode(self, x: Any) ‑> torch.Tensor`
    :   Encode data to the unimodal representation.
        Args:
            x: data of the domain.
        Returns:
            a unimodal representation.

    `on_before_gw_encode_broadcast(self, z: torch.Tensor) ‑> torch.Tensor`
    :

    `on_before_gw_encode_cont(self, z: torch.Tensor) ‑> torch.Tensor`
    :

    `on_before_gw_encode_cy(self, z: torch.Tensor) ‑> torch.Tensor`
    :

    `on_before_gw_encode_dcy(self, z: torch.Tensor) ‑> torch.Tensor`
    :

    `on_before_gw_encode_tr(self, z: torch.Tensor) ‑> torch.Tensor`
    :

`LossOutput(loss: torch.Tensor, metrics: dict[str, torch.Tensor] = <factory>)`
:   LossOutput(loss: torch.Tensor, metrics: dict[str, torch.Tensor] = <factory>)

    ### Class variables

    `loss: torch.Tensor`
    :

    `metrics: dict[str, torch.Tensor]`
    :

    ### Instance variables

    `all: dict[str, torch.Tensor]`
    :   Returns a dict with all metrics and loss with "loss" key