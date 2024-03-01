# DomainModule API

## `type shimmer.modules.losses.LatentsDomainGroupT`
```python
LatentsDomainGroupT = Mapping[str, torch.Tensor]
```
Matched unimodal latent representations from multiple domains.
Keys of the mapping are domains names.

## `type shimmer.modules.losses.LatentsT`
```python
LatentsT = Mapping[frozenset[str], LatentsDomainGroupT]
```
Mapping of `LatentsDomainGroupT`. Keys are frozenset of domains matched in the group.
Each group is independent and contains different data (unpaired).

## `class shimmer.modules.losses.GWLossesBase`
Base Abstract Class for Global Workspace (GW) losses. This module is used
to compute the different losses of the GW (typically translation, cycle,
demi-cycle, contrastive losses).

### step 
```python
@abstractmethod
step(domain_latents: LatentsT, mode: str) -> LossOutput
```
**Args**:
- **domain\_latents** (`LatentsT`) groups of matched unimodal latent representations.
- **mode** (`str`) train/val/test/predict

**Return type**: [`shimmer.modules.domain.LossOutput`](./loss_output.md)

## `class shimmer.modules.losses.LossCoefs`
TypedDict of loss coefficients used in the GlobalWorkspace
If one is not provided, the coefficient is assumed to be 0 and will not be logged.
If the loss is excplicitely set to 0, it will be logged, but not take part in
the total loss.

### demi_cycles
```python
LossCoefs.demi_cycles: float
```
Demi-cycle loss coefficient.

### cycles
```python
LossCoefs.cycles: float
```
Cycle loss coefficient.

### translations
```python
LossCoefs.translations: float
```
Translations loss coefficient.

### contrastives
```python
LossCoefs.contrastives: float
```
Contrastives loss coefficient.

## `class shimmer.modules.losses.GWLosses`
Implements `GWLossesBase`. This is the main loss module to use
with the GlobalWorkspace.

### \_\_init\_\_
```python
__init__(
    gw_mod: GWModule,
    domain_mods: dict[str, DomainModule],
    loss_coefs: LossCoefs,
    contrastive_fn: ContrastiveLossType,
) -> None
```
**Args**:
- **gw\_mod** (`GWModule`) groups of matched unimodal latent representations.
- **mode** (`str`) train/val/test/predict

**Return type**: [`shimmer.modules.domain.LossOutput`](./loss_output.md)
