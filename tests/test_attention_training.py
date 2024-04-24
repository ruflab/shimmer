import torch
import torch.nn as nn
from utils import DummyDataset, DummyDomainModule

from shimmer import GWDecoder, GWEncoder
from shimmer.modules.attention_module import ClassificationHead, DynamicAttention
from shimmer.modules.global_workspace import (
    GlobalWorkspaceFusion,
    SchedulerArgs,
)
from shimmer.modules.losses import BroadcastLossCoefs


class DummyCriterion(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.classification_head = ClassificationHead(
            input_dim=latent_dim, output_dim=3
        )

    def forward(self, input, target):
        # Calculate some dummy loss
        loss = torch.mean(input - target)  # Dummy loss calculation
        return loss


def test_attention_training():
    train_dataset = DummyDataset(
        size=128,
        domains=["v_latents", "attr"],
    )

    domains = {
        "v_latents": DummyDomainModule(latent_dim=128),
        "attr": DummyDomainModule(latent_dim=128),
    }

    workspace_dim = 16

    gw_encoders = {
        "v_latents": GWEncoder(
            domains["v_latents"].latent_dim,
            hidden_dim=64,
            out_dim=workspace_dim,
            n_layers=1,
        ),
        "attr": GWEncoder(
            domains["attr"].latent_dim,
            hidden_dim=64,
            out_dim=workspace_dim,
            n_layers=1,
        ),
    }

    gw_decoders = {
        "v_latents": GWDecoder(
            workspace_dim,
            hidden_dim=64,
            out_dim=domains["v_latents"].latent_dim,
            n_layers=1,
        ),
        "attr": GWDecoder(
            workspace_dim,
            hidden_dim=64,
            out_dim=domains["attr"].latent_dim,
            n_layers=1,
        ),
    }
    latent_dim = 12
    # loss_coefs_fusion: BroadcastLossCoefs = {
    #     "contrastives": 0.05,
    #     "broadcast": 1.0,
    # }
    loss_coefs_fusion: BroadcastLossCoefs = {"contrastives": 0.05}
    optim_lr = 0.005
    weight_decay = 1e-06

    module = GlobalWorkspaceFusion(
        domains,
        gw_encoders,
        gw_decoders,
        latent_dim,
        loss_coefs_fusion,
        optim_lr,
        weight_decay,
        scheduler_args=SchedulerArgs(
            max_lr=0.005,
            total_steps=100000,
        ),
        learn_logit_scale=False,
        contrastive_loss=None,
    )

    # Initialize attention mechanism
    domain_dim = 12
    head_size = 5
    batch_size = 2056
    domains = ("v_latents", "attr")

    criterion = DummyCriterion(domain_dim)
    optim_lr = 1e-3

    attention_module = DynamicAttention(
        module, batch_size, domain_dim, head_size, domains, criterion, optim_lr
    )

    # trainer = Trainer(
    #     logger=wandb_logger,
    #     fast_dev_run=config.training.fast_dev_run,
    #     max_steps=config.training.max_steps,
    #     enable_progress_bar=config.training.enable_progress_bar,
    #     default_root_dir=config.default_root_dir,
    #     callbacks=callbacks,
    #     precision=config.training.precision,
    #     accelerator=config.training.accelerator,
    #     devices=config.training.devices,
    # )
    # trainer.fit(attention_module, train_dataloader)
    # # Test forward pass

    # # Test apply corruption

    # train_datalader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # batch = next(iter(train_datalader))

    # assert batch.keys() == {"v", "t"}
    # assert isinstance(batch["v"], DummyData)

    # unimodal_latents: dict[str, torch.Tensor] = {
    #     domain: domains[domain].encode(batch[domain]) for domain in batch
    # }
    # assert isinstance(unimodal_latents["v"], torch.Tensor)
    # assert unimodal_latents["v"].size() == (32, 128)

    # selection_module = SingleDomainSelection()
    # workspace_latent = gw.gw_mod.encode_and_fuse(unimodal_latents, selection_module)

    # assert workspace_latent.size() == (32, 16)

    # reconstructed_unimodal_latents = gw.gw_mod.decode(
    #     workspace_latent, domains={"v", "a"}
    # )

    # assert reconstructed_unimodal_latents.keys() == {"v", "a"}
    # assert reconstructed_unimodal_latents["v"].size() == (32, 128)
