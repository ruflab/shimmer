# LossOutput

## `class shimmer.modules.domain.LossOutput`
This is a python dataclass use as a returned value for losses.
It keeps track of what is used for training (`loss`) and what is used
only for logging (`metrics`)

## \_\_init\_\_
```python
__init__(loss: torch.Tensor, metrics: dict[str, torch.Tensor]) -> None
```
Initializes LossOutput. Note that `metrics` cannot have the key "loss".

**Args**:
- **loss** (`torch.Tensor`) the loss to train on
- **metrics** (`dict[str, torch.Tensor]`) a dict of additional metrics to send to the
logger.

**Return type**: `None`


## loss
```python
LossOutput.loss: torch.Tensor
```
The loss value.

## metrics
```python
LossOutput.metrics: dict[str, torch.Tensor]
```
The metrics dict.

## all
```python
LossOutput.metrics: dict[str, torch.Tensor]
```
A dict that combines `LossOutput.metrics` and `LossOutput.loss` where the loss is
under the key "loss".
