Module shimmer.modules.gw_module
================================

Functions
---------

    
`get_n_layers(n_layers: int, hidden_dim: int)`
:   

Classes
-------

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

`GWModuleFusion(gw_interfaces: collections.abc.Mapping[str, shimmer.modules.gw_module.GWInterfaceBase], workspace_dim: int)`
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

    * shimmer.modules.gw_module.GWModule
    * shimmer.modules.gw_module.GWModuleBase
    * torch.nn.modules.module.Module
    * abc.ABC

    ### Methods

    `get_batch_size(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> int`
    :

    `get_device(self, x: collections.abc.Mapping[str, torch.Tensor]) ‑> torch.device`
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