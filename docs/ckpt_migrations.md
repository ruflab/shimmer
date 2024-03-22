# Checkpoint Migrations

When changin shimmer versions, changes made in the lib may break your old checkpoints.
To avoid this, we provide migration functions that can update checkpoints so that
they can be used with newer versions.

The repo for code migration is in: https://github.com/bdvllrs/migrate-ckpt

## Manually migrate checkpoints

If you install shimmer with pip:
```sh
shimmer migrate-ckpt path/to/checkpoint1.ckpt checkpoint2.ckpt ...
```

## Migrate checkpoint in your own script

```python
from shimmer import migrate_model


migrate_model()
```



## Automatically save existing migrations when creating a new checkpoint

You can use the provided lightning callback that will add the existing migration
in the checkpoint when saving the checkpoint.

```python
from lightning.pytorch import Trainer
from shimmer import SaveMigrations

callback = SaveMigrations()

trainer = Trainer(..., callbacks=[callback])
...
```
