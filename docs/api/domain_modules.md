# DomainModule API

## class shimmer.modules.domain.DomainModule
DomainModules define the domain specific module of a Global Workspace (GW).


### \_\_init\_\_(latent\_dim: int) -> None
Initialize a DomainModule

**Args:**
- **latent\_dim**(int) latent dimension of the unimodal module

**Return type**: None

### encode(x: Any) -> torch.Tensor
Encode the domain data into a unimodal latent representation.

**Args:**

- **x**(Any) input data

**Return type**: torch.Tensor

### decode(z: torch.Tensor) -> Any
Decode the unimodal representation back to the initial domain data.

This should be true:
DomainModule.decode(DomainModule.encode(x)) == x
**Args:**

- **z**(torch.Tensor) latent representation

**Return type**: Any

### compute\_loss(pred: torch.Tensor, target: torch.Tensor) -> LossOutput
Generic loss used for translation, demi-cycle and cycle losses for this domain.

**Args:**

- **pred**(torch.Tensor) prediction of the model
- **target**(torch.Tensor) target tensor

**Return type**: shimmer.modules.domain.LossOutput

### on\_before\_gw\_encode\_dcy(z: torch.Tensor) -> torch.Tensor
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a demi-cycle loss.

If not defined, will return the input (identity function).

**Args:**

- **z**(torch.Tensor) latent representation

**Return type**: torch.Tensor 
**Returns**: updated latent representation

### on\_before\_gw\_encode\_cont(z: torch.Tensor) -> torch.Tensor
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a contrastive loss.

If not defined, will return the input (identity function).

**Args:**

- **z**(torch.Tensor) latent representation

**Return type**: torch.Tensor 
**Returns**: updated latent representation

### on\_before\_gw\_encode\_tr(z: torch.Tensor) -> torch.Tensor
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a translation loss.

If not defined, will return the input (identity function).

**Args:**

- **z**(torch.Tensor) latent representation

**Return type**: torch.Tensor 
**Returns**: updated latent representation

### on\_before\_gw\_encode\_cy(z: torch.Tensor) -> torch.Tensor
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a cycle loss.

If not defined, will return the input (identity function).

**Args:**

- **z**(torch.Tensor) latent representation

**Return type**: torch.Tensor 
**Returns**: updated latent representation

### on\_before\_gw\_encode\_broadcast(z: torch.Tensor) -> torch.Tensor
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a broadcast loss (for fusion GW).

If not defined, will return the input (identity function).

**Args:**

- **z**(torch.Tensor) latent representation

**Return type**: torch.Tensor 
**Returns**: updated latent representation
