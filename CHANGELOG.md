# 0.1.0
Initial release

# 0.1.1
Fix missing individual metrics for translation loss.

# 0.1.2
Fix wrong module used to compute the cycle losses. Don't do cycle with the same domain as target and source.

# 0.2.0
Add callback on\_before\_gw\_encode and individual compute\_losses for each loss type.
Fix bugs

# 0.3.0
* Breaking change: remove `DeterministGlobaleWorkspace` and `VariationalGlobalWorkspace`
in favor of the functions: `global_workspace` and `variational_global_workspace`.
* Allow setting custom GW encoders and decoders.
* Breaking change: remove `self.input_dim`, `self.encoder_hidden_dim`, 
`self.encoder_n_layers`, `self.decoder_hidden_dim`, and `self.decoder_n_layers`
in `GWModule`s.

# 0.3.1
Fix bugs related to imports and `default_decoders`.

# 0.3.2
* Revert to using classes for GWs (it's easier when loading from checkpoints.)

* `GlobalWorkspace` is renamed to `GlobalWorkspaceBase` and `GlobalWorkspace` now
refers to `DeterministicGlobalWorkspace`.

# 0.4.0
* Use ABC for abstract methods.
* Replace `DomainDescription` with `GWInterface`.
* Add `contrastive_fn` attribute in `DeterministicGWLosses` to compute the contrastive loss.
    It can then be customized.
* Rename every abstract class with ClassNameBase. Rename every "Deterministic" classes 
    to remove "Deterministic".
* Remove all config related functions. This is not the role of this repo.

# 0.5.0
* Replace loss coef buffers by a `LossCoef` TypedDict.
* Add
  [`RepeatedDataset`](https://bdvllrs.github.io/shimmer/shimmer/dataset.html#RepeatedDataset)
  to shimmer.
* Add docs in `docs/`, API documentation in https://bdvllrs.github.io/shimmer/, and
    some code examples.
* Replace Black, isort, and flake8 with Ruff (see
      https://github.com/bdvllrs/shimmer/pull/8).
* Remove `GWInterfaces` entirely and favor giving encoders and decoders directly to the
    `GWModule`. See the updated example `examples/main_example/train_gw.py` to see what 
    changes to make (see https://github.com/bdvllrs/shimmer/pull/9).
* Remove `GWModuleBase.translate`  and `GWModuleBase.cycle`. Translation and cycles
    can now be done with the utils function `translation` and `cycle`.
* Remove `GlobalWorkspaceBase.batch_demi_cycles`, `GlobalWorkspaceBase.batch_cycles`, 
    and `GlobalWorkspaceBase.batch_translations`. This can be done with utils
    functions of the same name.
* Rename `GWModuleBase.fusion_mechanism` to `GWModuleBase.fuse`,
    `GWModuleBase.encode` to `GWModuleBase.encode_and_fuse`, and
    `GWModuleBase.encode_pre_fusion` to `GWModuleBase.encode`. Same for the associated
    methods in `GlobalWorkspaceBase`.
* Remove on_before_gw_encode_{loss} callbacks to allow sharing computation between
    loss functions.
* Remove many _with_uncertainty functions. The GWModuleWithUncertainty now behaves like
    the other GWModules.
* Rename all "with_uncertainty" stuff to "bayesian"
