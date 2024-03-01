Module shimmer.modules.vae
==========================

Functions
---------

    
`gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, x: torch.Tensor) ‑> torch.Tensor`
:   

    
`kl_divergence_loss(mean: torch.Tensor, logvar: torch.Tensor) ‑> torch.Tensor`
:   

    
`reparameterize(mean: torch.Tensor, logvar: torch.Tensor) ‑> torch.Tensor`
:   

Classes
-------

`VAE(encoder: shimmer.modules.vae.VAEEncoder, decoder: shimmer.modules.vae.VAEDecoder, beta: float = 1)`
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

    `decode(self, z: torch.Tensor) ‑> collections.abc.Sequence[torch.Tensor]`
    :

    `encode(self, x: collections.abc.Sequence[torch.Tensor]) ‑> torch.Tensor`
    :

    `forward(self, x: collections.abc.Sequence[torch.Tensor]) ‑> tuple[tuple[torch.Tensor, torch.Tensor], collections.abc.Sequence[torch.Tensor]]`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`VAEDecoder(*args, **kwargs)`
:   Base class for a VAE decoder.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module
    * abc.ABC

    ### Methods

    `forward(self, x: torch.Tensor) ‑> collections.abc.Sequence[torch.Tensor]`
    :   Decode representation with VAE
        Args:
            x: representation
        Retunrs:
            Sequence of tensors reconstructing input

`VAEEncoder(*args, **kwargs)`
:   Base class for a VAE encoder.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module
    * abc.ABC

    ### Methods

    `forward(self, x: collections.abc.Sequence[torch.Tensor]) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Encode representation with VAE
        Args:
            x: sequence of tensors
        Retunrs:
            tuple with the mean and the log variance