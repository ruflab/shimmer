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
