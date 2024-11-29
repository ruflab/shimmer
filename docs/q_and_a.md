# Q&A

## How can I customize my GlobalWorkspace
[`GlobalWorkspaceBase`](https://ruflab.github.io/shimmer/latest/shimmer/modules/global_workspace.html#GlobalWorkspaceBase) 
is a very generic implementation of the global workspace and uses different building
blocks (given as arguments) to function:
* **gw_mod**: a class implementation of `GWModuleBase` which defines how to encode, 
    decode, and fuse domains.
* **loss_mod**: a class implementation of `GWLossesBase` which defines computes and
    defines losses to train on, and metrics to log.
* aditionnal optimization parameters (see API docs).

Different implementations have been made to initialize the Global Workspace more easily,
but you may need to implement a new one if you have very specific needs.

To get insipiration, you can look at the source code of
[`GlobalWorkspace`](https://ruflab.github.io/shimmer/latest/shimmer/modules/global_workspace.html#GlobalWorkspace).

## How can I change the loss function?
If you are using pre-made GW architecture
([`GlobalWorkspace2Domains`](https://ruflab.github.io/shimmer/latest/shimmer/modules/global_workspace.html#GlobalWorkspace2Domains),
[`GlobalWorkspaceFusion`](https://ruflab.github.io/shimmer/latest/shimmer/modules/global_workspace.html#GlobalWorkspaceFusion)) and want to update the loss
used for demi-cycles, cycles, translations or broadcast, you can do so directly from
your definition of the
[`DomainModule`](https://ruflab.github.io/shimmer/latest/shimmer/modules/domain.html#DomainModule.compute_loss)
when defining the `compute_loss` method.

You also can have different losses for demi-cycles, cycles, ... by implementing
the corresponding methods
([`compute_dcy_loss`](https://ruflab.github.io/shimmer/latest/shimmer/modules/domain.html#DomainModule.compute_dcy_loss),
[`compute_tr_loss`](https://ruflab.github.io/shimmer/latest/shimmer/modules/domain.html#DomainModule.compute_tr_loss), ...).

You can use your own contrastive loss function by passing it as an argument to the
`GlobalWorkspace` class with the `contrastive_loss` loss argument.

If you have more specific needs, like change how the different loss interact, or
completely replace the loss combination we provide, you will need to implement a new
Loss Module inheriting from
[`GWLossesBase`](https://ruflab.github.io/shimmer/latest/shimmer/modules/losses.html#GWLossesBase).
Then, you can create a new implementation of
[`GlobalWorkspaceBase`](https://ruflab.github.io/shimmer/latest/shimmer/modules/global_workspace.html#GlobalWorkspaceBase)
(see section "How can I customize my GlobalWorkspace" for more details.)


## shimmer has broken my checkpoints! How can I upgrade them?
See [Checkpoint Migrations](ckpt_migrations.md).
