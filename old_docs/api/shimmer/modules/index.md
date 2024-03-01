Module shimmer.modules
======================

Sub-modules
-----------
* shimmer.modules.domain
* shimmer.modules.global_workspace
* shimmer.modules.gw_module
* shimmer.modules.losses
* shimmer.modules.vae

Functions
---------

    
`contrastive_loss(x: torch.Tensor, y: torch.Tensor, logit_scale: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean') ‑> torch.Tensor`
:   

    
`contrastive_loss_with_uncertainty(x: torch.Tensor, x_log_uncertainty: torch.Tensor, y: torch.Tensor, y_log_uncertainty: torch.Tensor, logit_scale: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean') ‑> torch.Tensor`
:   

    
`pretrained_global_workspace(checkpoint_path: str | pathlib.Path, domain_mods: collections.abc.Mapping[str, shimmer.modules.domain.DomainModule], gw_interfaces: collections.abc.Mapping[str, shimmer.modules.gw_module.GWInterfaceBase], workspace_dim: int, loss_coefs: shimmer.modules.losses.LossCoefs, contrastive_fn: collections.abc.Callable[[torch.Tensor, torch.Tensor], shimmer.modules.domain.LossOutput], **kwargs) ‑> shimmer.modules.global_workspace.GlobalWorkspace`
:   

Classes
-------

`ContrastiveLoss(logit_scale: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean', learn_logit_scale: bool = False)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, y: torch.Tensor) ‑> shimmer.modules.domain.LossOutput`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`ContrastiveLossWithUncertainty(logit_scale: torch.Tensor, reduction: Literal['mean', 'sum', 'none'] = 'mean', learn_logit_scale: bool = False)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    ContrastiveLoss used for VariationalGlobalWorkspace

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, x_log_uncertainty: torch.Tensor, y: torch.Tensor, y_log_uncertainty: torch.Tensor) ‑> shimmer.modules.domain.LossOutput`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

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

`GWDecoder(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int)`
:   A sequential container.
    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.
    
    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).
    
    What's the difference between a ``Sequential`` and a
    :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
    sounds like--a list for storing ``Module`` s! On the other hand,
    the layers in a ``Sequential`` are connected in a cascading way.
    
    Example::
    
        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
    
        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.container.Sequential
    * torch.nn.modules.module.Module

    ### Descendants

    * shimmer.modules.gw_module.GWEncoder

`GWEncoder(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int)`
:   A sequential container.
    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.
    
    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).
    
    What's the difference between a ``Sequential`` and a
    :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
    sounds like--a list for storing ``Module`` s! On the other hand,
    the layers in a ``Sequential`` are connected in a cascading way.
    
    Example::
    
        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
    
        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * shimmer.modules.gw_module.GWDecoder
    * torch.nn.modules.container.Sequential
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`GWInterface(domain_module: shimmer.modules.domain.DomainModule, workspace_dim: int, encoder_hidden_dim: int, encoder_n_layers: int, decoder_hidden_dim: int, decoder_n_layers: int)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * shimmer.modules.gw_module.GWInterfaceBase
    * torch.nn.modules.module.Module
    * abc.ABC

    ### Methods

    `decode(self, z: torch.Tensor) ‑> torch.Tensor`
    :

    `encode(self, x: torch.Tensor) ‑> torch.Tensor`
    :

`GWInterfaceBase(domain_module: shimmer.modules.domain.DomainModule, workspace_dim: int)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module
    * abc.ABC

    ### Descendants

    * shimmer.modules.gw_module.GWInterface
    * shimmer.modules.gw_module.VariationalGWInterface

    ### Methods

    `decode(self, z: torch.Tensor) ‑> torch.Tensor`
    :

    `encode(self, x: torch.Tensor) ‑> torch.Tensor`
    :

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

`GWModule(gw_interfaces: collections.abc.Mapping[str, shimmer.modules.gw_module.GWInterfaceBase], workspace_dim: int)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * shimmer.modules.gw_module.GWModuleBase
    * torch.nn.modules.module.Module
    * abc.ABC

    ### Descendants

    * shimmer.modules.gw_module.GWModuleFusion

    ### Methods

    `fusion_mechanism(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> torch.Tensor`
    :   Merge function used to combine domains.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            The merged representation

`GWModuleBase(gw_interfaces: collections.abc.Mapping[str, shimmer.modules.gw_module.GWInterfaceBase], workspace_dim: int)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module
    * abc.ABC

    ### Descendants

    * shimmer.modules.gw_module.GWModule
    * shimmer.modules.gw_module.VariationalGWModule

    ### Methods

    `cycle(self, x: collections.abc.Mapping[str, torch.Tensor], through: str) ‑> dict[str, torch.Tensor]`
    :   Cycle from one domain through another.
        Args:
            x: mapping of domain name to unimodal representation.
            through: domain to translate to.
        Returns:
            the unimodal representations cycles through the given domain.

    `decode(self, z: torch.Tensor, domains: collections.abc.Iterable[str] | None = None) ‑> dict[str, torch.Tensor]`
    :   Decode the GW representation to the unimodal representations.
        Args:
            z: GW representation
            domains: iterable of domains to decode to. Defaults to all domains.
        Returns:
            dict of domain name to decoded unimodal representation.

    `encode(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> torch.Tensor`
    :   Encode the unimodal representations to the GW representation.
        Args:
            x: mapping of domain name to unimodal representation.
        Returns:
            GW representation

    `on_before_gw_encode_cont(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> dict[str, torch.Tensor]`
    :   Callback used before projecting the unimodal representations to the GW
        representation when computing the contrastive loss. Defaults to identity.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            the same mapping with updated representations

    `on_before_gw_encode_cy(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> dict[str, torch.Tensor]`
    :   Callback used before projecting the unimodal representations to the GW
        representation when computing the cycle loss. Defaults to identity.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            the same mapping with updated representations

    `on_before_gw_encode_dcy(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> dict[str, torch.Tensor]`
    :   Callback used before projecting the unimodal representations to the GW
        representation when computing the demi-cycle loss. Defaults to identity.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            the same mapping with updated representations

    `on_before_gw_encode_tr(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> dict[str, torch.Tensor]`
    :   Callback used before projecting the unimodal representations to the GW
        representation when computing the translation loss. Defaults to identity.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            the same mapping with updated representations

    `translate(self, x: collections.abc.Mapping[str, torch.Tensor], to: str) ‑> torch.Tensor`
    :   Translate from one domain to another.
        Args:
            x: mapping of domain name to unimodal representation.
            to: domain to translate to.
        Returns:
            the unimodal representation of domain given by `to`.

`GlobalWorkspace(domain_mods: collections.abc.Mapping[str, shimmer.modules.domain.DomainModule], gw_interfaces: collections.abc.Mapping[str, shimmer.modules.gw_module.GWInterfaceBase], workspace_dim: int, loss_coefs: shimmer.modules.losses.LossCoefs, optim_lr: float = 0.001, optim_weight_decay: float = 0.0, scheduler_args: shimmer.modules.global_workspace.SchedulerArgs | None = None, learn_logit_scale: bool = False, contrastive_loss: collections.abc.Callable[[torch.Tensor, torch.Tensor], shimmer.modules.domain.LossOutput] | None = None)`
:   Hooks to be used in LightningModule.

    ### Ancestors (in MRO)

    * shimmer.modules.global_workspace.GlobalWorkspaceBase
    * lightning.pytorch.core.module.LightningModule
    * lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    * lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    * lightning.pytorch.core.hooks.ModelHooks
    * lightning.pytorch.core.hooks.DataHooks
    * lightning.pytorch.core.hooks.CheckpointHooks
    * torch.nn.modules.module.Module

`GlobalWorkspaceBase(gw_mod: shimmer.modules.gw_module.GWModuleBase, domain_mods: collections.abc.Mapping[str, shimmer.modules.domain.DomainModule], loss_mod: shimmer.modules.losses.GWLossesBase, optim_lr: float = 0.001, optim_weight_decay: float = 0.0, scheduler_args: shimmer.modules.global_workspace.SchedulerArgs | None = None)`
:   Hooks to be used in LightningModule.

    ### Ancestors (in MRO)

    * lightning.pytorch.core.module.LightningModule
    * lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    * lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    * lightning.pytorch.core.hooks.ModelHooks
    * lightning.pytorch.core.hooks.DataHooks
    * lightning.pytorch.core.hooks.CheckpointHooks
    * torch.nn.modules.module.Module

    ### Descendants

    * shimmer.modules.global_workspace.GlobalWorkspace
    * shimmer.modules.global_workspace.GlobalWorkspaceFusion
    * shimmer.modules.global_workspace.VariationalGlobalWorkspace

    ### Instance variables

    `workspace_dim`
    :

    ### Methods

    `batch_cycles(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[tuple[str, str], torch.Tensor]`
    :

    `batch_demi_cycles(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `batch_gw_states(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[str, torch.Tensor]`
    :

    `batch_translations(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[tuple[str, str], torch.Tensor]`
    :

    `configure_optimizers(self) ‑> lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig`
    :   Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you'd need one.
        But in the case of GANs or similar you might have multiple. Optimization with multiple optimizers only works in
        the manual optimization mode.
        
        Return:
            Any of these 6 options.
        
            - **Single optimizer**.
            - **List or Tuple** of optimizers.
            - **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers
              (or multiple ``lr_scheduler_config``).
            - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
              key whose value is a single LR scheduler or ``lr_scheduler_config``.
            - **None** - Fit will run without any optimizer.
        
        The ``lr_scheduler_config`` is a dictionary which contains the scheduler and its associated configuration.
        The default configuration is shown below.
        
        .. code-block:: python
        
            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }
        
        When there are schedulers in which the ``.step()`` method is conditioned on a value, such as the
        :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler, Lightning requires that the
        ``lr_scheduler_config`` contains the keyword ``"monitor"`` set to the metric name that the scheduler
        should be conditioned on.
        
        .. testcode::
        
            # The ReduceLROnPlateau scheduler requires a monitor
            def configure_optimizers(self):
                optimizer = Adam(...)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(optimizer, ...),
                        "monitor": "metric_to_track",
                        "frequency": "indicates how often the metric is updated",
                        # If "monitor" references validation metrics, then "frequency" should be set to a
                        # multiple of "trainer.check_val_every_n_epoch".
                    },
                }
        
        
            # In the case of two optimizers, only one using the ReduceLROnPlateau scheduler
            def configure_optimizers(self):
                optimizer1 = Adam(...)
                optimizer2 = SGD(...)
                scheduler1 = ReduceLROnPlateau(optimizer1, ...)
                scheduler2 = LambdaLR(optimizer2, ...)
                return (
                    {
                        "optimizer": optimizer1,
                        "lr_scheduler": {
                            "scheduler": scheduler1,
                            "monitor": "metric_to_track",
                        },
                    },
                    {"optimizer": optimizer2, "lr_scheduler": scheduler2},
                )
        
        Metrics can be made available to monitor by simply logging it using
        ``self.log('metric_to_track', metric_val)`` in your :class:`~lightning.pytorch.core.LightningModule`.
        
        Note:
            Some things to know:
        
            - Lightning calls ``.backward()`` and ``.step()`` automatically in case of automatic optimization.
            - If a learning rate scheduler is specified in ``configure_optimizers()`` with key
              ``"interval"`` (default "epoch") in the scheduler configuration, Lightning will call
              the scheduler's ``.step()`` method automatically in case of automatic optimization.
            - If you use 16-bit precision (``precision=16``), Lightning will automatically handle the optimizer.
            - If you use :class:`torch.optim.LBFGS`, Lightning handles the closure function automatically for you.
            - If you use multiple optimizers, you will have to switch to 'manual optimization' mode and step them
              yourself.
            - If you need to control how often the optimizer steps, override the :meth:`optimizer_step` hook.

    `decode(self, z: torch.Tensor, domains: collections.abc.Iterable[str] | None = None) ‑> dict[str, torch.Tensor]`
    :

    `decode_domain(self, domain: torch.Tensor, name: str) ‑> Any`
    :

    `decode_domains(self, latents_domain: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> dict[frozenset[str], dict[str, typing.Any]]`
    :

    `encode(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> torch.Tensor`
    :

    `encode_domain(self, domain: Any, name: str) ‑> torch.Tensor`
    :

    `encode_domains(self, batch: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, typing.Any]]) ‑> dict[frozenset[str], dict[str, torch.Tensor]]`
    :

    `forward(self, latent_domains: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, torch.Tensor]]) ‑> shimmer.modules.global_workspace.GWPredictions`
    :   Same as :meth:`torch.nn.Module.forward`.
        
        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.
        
        Return:
            Your model's output

    `generic_step(self, batch: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, typing.Any]], mode: str) ‑> torch.Tensor`
    :

    `predict_step(self, data: collections.abc.Mapping[str, typing.Any], _) ‑> shimmer.modules.global_workspace.GWPredictions`
    :   Step function called during :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`. By default, it calls
        :meth:`~lightning.pytorch.core.LightningModule.forward`. Override to add any processing logic.
        
        The :meth:`~lightning.pytorch.core.LightningModule.predict_step` is used
        to scale inference on multi-devices.
        
        To prevent an OOM error, it is possible to use :class:`~lightning.pytorch.callbacks.BasePredictionWriter`
        callback to write the predictions to disk or database after each batch or on epoch end.
        
        The :class:`~lightning.pytorch.callbacks.BasePredictionWriter` should be used while using a spawn
        based accelerator. This happens for ``Trainer(strategy="ddp_spawn")``
        or training on 8 TPU cores with ``Trainer(accelerator="tpu", devices=8)`` as predictions won't be returned.
        
        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)
        
        Return:
            Predicted output (optional).
        
        Example ::
        
            class MyModel(LightningModule):
        
                def predict_step(self, batch, batch_idx, dataloader_idx=0):
                    return self(batch)
        
            dm = ...
            model = MyModel()
            trainer = Trainer(accelerator="gpu", devices=2)
            predictions = trainer.predict(model, dm)

    `test_step(self, data: collections.abc.Mapping[str, typing.Any], _, dataloader_idx: int = 0) ‑> torch.Tensor`
    :   Operates on a single batch of data from the test set. In this step you'd normally generate examples or
        calculate anything of interest such as accuracy.
        
        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)
        
        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch.
        
        .. code-block:: python
        
            # if you have one test dataloader:
            def test_step(self, batch, batch_idx): ...
        
        
            # if you have multiple test dataloaders:
            def test_step(self, batch, batch_idx, dataloader_idx=0): ...
        
        Examples::
        
            # CASE 1: A single test dataset
            def test_step(self, batch, batch_idx):
                x, y = batch
        
                # implement your own
                out = self(x)
                loss = self.loss(out, y)
        
                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)
        
                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        
                # log the outputs!
                self.log_dict({'test_loss': loss, 'test_acc': test_acc})
        
        If you pass in multiple test dataloaders, :meth:`test_step` will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.
        
        .. code-block:: python
        
            # CASE 2: multiple test dataloaders
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...
        
        Note:
            If you don't need to test you don't need to implement this method.
        
        Note:
            When the :meth:`test_step` is called, the model has been put in eval mode and
            PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
            to training mode and gradients are enabled.

    `training_step(self, batch: collections.abc.Mapping[frozenset[str], collections.abc.Mapping[str, typing.Any]], _) ‑> torch.Tensor`
    :   Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.
        
        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)
        
        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary which can include any keys, but must include the key ``'loss'`` in the case of
              automatic optimization.
            - ``None`` - In automatic optimization, this will skip to the next batch (but is not supported for
              multi-GPU, TPU, or DeepSpeed). For manual optimization, this has no special meaning, as returning
              the loss is not required.
        
        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.
        
        Example::
        
            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss
        
        To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:
        
        .. code-block:: python
        
            def __init__(self):
                super().__init__()
                self.automatic_optimization = False
        
        
            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()
        
                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()
        
        Note:
            When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
            normalized by ``accumulate_grad_batches`` internally.

    `validation_step(self, data: collections.abc.Mapping[str, typing.Any], _, dataloader_idx: int = 0) ‑> torch.Tensor`
    :   Operates on a single batch of data from the validation set. In this step you'd might generate examples or
        calculate anything of interest like accuracy.
        
        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)
        
        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch.
        
        .. code-block:: python
        
            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx): ...
        
        
            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idx=0): ...
        
        Examples::
        
            # CASE 1: A single validation dataset
            def validation_step(self, batch, batch_idx):
                x, y = batch
        
                # implement your own
                out = self(x)
                loss = self.loss(out, y)
        
                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)
        
                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        
                # log the outputs!
                self.log_dict({'val_loss': loss, 'val_acc': val_acc})
        
        If you pass in multiple val dataloaders, :meth:`validation_step` will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.
        
        .. code-block:: python
        
            # CASE 2: multiple validation dataloaders
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...
        
        Note:
            If you don't need to validate you don't need to implement this method.
        
        Note:
            When the :meth:`validation_step` is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of validation,
            the model goes back to training mode and gradients are enabled.

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

`SchedulerArgs(*args, **kwargs)`
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

    `max_lr: float`
    :

    `total_steps: int`
    :

`VariationalGWEncoder(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`VariationalGWInterface(domain_module: shimmer.modules.domain.DomainModule, workspace_dim: int, encoder_hidden_dim: int, encoder_n_layers: int, decoder_hidden_dim: int, decoder_n_layers: int)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * shimmer.modules.gw_module.GWInterfaceBase
    * torch.nn.modules.module.Module
    * abc.ABC

    ### Methods

    `decode(self, z: torch.Tensor) ‑> torch.Tensor`
    :

    `encode(self, x: torch.Tensor) ‑> torch.Tensor`
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

`VariationalGWModule(gw_interfaces: collections.abc.Mapping[str, shimmer.modules.gw_module.GWInterfaceBase], workspace_dim: int)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * shimmer.modules.gw_module.GWModuleBase
    * torch.nn.modules.module.Module
    * abc.ABC

    ### Methods

    `encode_mean(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> torch.Tensor`
    :

    `encoded_distribution(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]`
    :

    `fusion_mechanism(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> torch.Tensor`
    :

`VariationalGlobalWorkspace(domain_mods: collections.abc.Mapping[str, shimmer.modules.domain.DomainModule], gw_interfaces: collections.abc.Mapping[str, shimmer.modules.gw_module.GWInterfaceBase], workspace_dim: int, loss_coefs: shimmer.modules.losses.VariationalLossCoefs, use_var_contrastive_loss: bool = False, optim_lr: float = 0.001, optim_weight_decay: float = 0.0, scheduler_args: shimmer.modules.global_workspace.SchedulerArgs | None = None, learn_logit_scale: bool = False, contrastive_loss: collections.abc.Callable[[torch.Tensor, torch.Tensor], shimmer.modules.domain.LossOutput] | None = None, var_contrastive_loss: collections.abc.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], shimmer.modules.domain.LossOutput] | None = None)`
:   Hooks to be used in LightningModule.

    ### Ancestors (in MRO)

    * shimmer.modules.global_workspace.GlobalWorkspaceBase
    * lightning.pytorch.core.module.LightningModule
    * lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    * lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    * lightning.pytorch.core.hooks.ModelHooks
    * lightning.pytorch.core.hooks.DataHooks
    * lightning.pytorch.core.hooks.CheckpointHooks
    * torch.nn.modules.module.Module

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