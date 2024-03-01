# DomainModule API

## `class shimmer.modules.domain.DomainModule`
DomainModules define the domain specific module of a Global Workspace (GW).

## \_\_init\_\_
```python
__init__(latent_dim: int) -> None
```
Initialize a DomainModule

**Args**:
- **latent\_dim** (`int`) latent dimension of the unimodal module

**Return type**: `None`

## latent\_dim
```python
DomainModule.latent_dim: int
```
The latent dimension of the module.

# encode
```python
encode(x: Any) -> torch.Tensor
```
Encode the domain data into a unimodal latent representation.

**Args**:

- **x** (`Any`) input data

**Return type**: `torch.Tensor`

# decode
```python
decode(z: torch.Tensor) -> Any
```
Decode the unimodal representation back to the initial domain data.

This should be true:
DomainModule.decode(DomainModule.encode(x)) == x
**Args**:

- **z** (`torch.Tensor`) latent representation

**Return type**: `Any`

# compute\_loss
```python
compute_loss(pred: torch.Tensor, target: torch.Tensor) -> LossOutput
```
Generic loss used for translation, demi-cycle, cycle or broadcast losses for this domain.

**Args**:

- **pred** (`torch.Tensor`) prediction of the model
- **target** (`torch.Tensor`) target tensor

**Return type**: [`shimmer.modules.domain.LossOutput`](./loss_output.md)

# compute\_dcy\_loss
```python
compute_dcy_loss(pred: torch.Tensor, target: torch.Tensor) -> LossOutput
```
Demi-cycle loss for this domain. By default, uses `compute_loss`.

**Args**:

- **pred** (`torch.Tensor`) prediction of the model
- **target** (`torch.Tensor`) target tensor

**Return type**: [`shimmer.modules.domain.LossOutput`](./loss_output.md)

# compute\_cy\_loss
```python
compute_cy_loss(pred: torch.Tensor, target: torch.Tensor) -> LossOutput
```
Cycle loss for this domain. By default, uses `compute_loss`.

**Args**:

- **pred** (`torch.Tensor`) prediction of the model
- **target** (`torch.Tensor`) target tensor

**Return type**: [`shimmer.modules.domain.LossOutput`](./loss_output.md)

# compute\_tr\_loss
```python
compute_tr_loss(pred: torch.Tensor, target: torch.Tensor) -> LossOutput
```
Translation loss for this domain. By default, uses `compute_loss`.

**Args**:

- **pred** (`torch.Tensor`) prediction of the model
- **target** (`torch.Tensor`) target tensor

**Return type**: [`shimmer.modules.domain.LossOutput`](./loss_output.md)

# compute\_broadcast\_loss
```python
compute_broadcast_loss(pred: torch.Tensor, target: torch.Tensor) -> LossOutput
```
Broadcast loss for this domain. By default, uses `compute_loss`.
This loss is used when using the FusionGlobalWorkspace and replaces the other
losses.

**Args**:

- **pred** (`torch.Tensor`) prediction of the model
- **target** (`torch.Tensor`) target tensor

**Return type**: [`shimmer.modules.domain.LossOutput`](./loss_output.md)

# on\_before\_gw\_encode\_dcy
```python
on_before_gw_encode_dcy(z: torch.Tensor) -> torch.Tensor
```
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a demi-cycle loss.

If not defined, will return the input (identity function).

**Args**:

- **z** (`torch.Tensor`) latent representation

**Return type**: `torch.Tensor`
**Returns**: updated latent representation

# on\_before\_gw\_encode\_cont
```python
on_before_gw_encode_cont(z: torch.Tensor) -> torch.Tensor
```
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a contrastive loss.

If not defined, will return the input (identity function).

**Args**:

- **z** (`torch.Tensor`) latent representation

**Return type**: `torch.Tensor`
**Returns**: updated latent representation

# on\_before\_gw\_encode\_tr
```python
on_before_gw_encode_tr(z: torch.Tensor) -> torch.Tensor
```
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a translation loss.

If not defined, will return the input (identity function).

**Args**:

- **z** (`torch.Tensor`) latent representation

**Return type**: `torch.Tensor`
**Returns**: updated latent representation

# on\_before\_gw\_encode\_cy
```python
on_before_gw_encode_cy(z: torch.Tensor) -> torch.Tensor
```
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a cycle loss.

If not defined, will return the input (identity function).

**Args**:

- **z** (`torch.Tensor`) latent representation

**Return type**: `torch.Tensor`
**Returns**: updated latent representation

# on\_before\_gw\_encode\_broadcast
```python
on_before_gw_encode_broadcast(z: torch.Tensor) -> torch.Tensor
```
Some additional computation to do before encoding the unimodal latent representation
to the GW when doing a broadcast loss (for fusion GW).

If not defined, will return the input (identity function).

**Args**:

- **z** (`torch.Tensor`) latent representation

**Return type**: `torch.Tensor`
**Returns**: updated latent representation

