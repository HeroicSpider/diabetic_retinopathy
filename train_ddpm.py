import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.utils as vutils
import copy
import os
from pathlib import Path

from ddpm_unet import ClassConditionalUNet
from ddpm_scheduler import CosineNoiseScheduler


class EMACallback(pl.Callback):
    def __init__(self, decay=0.9999, backup_dir=None, backup_every_n_epochs=25):
        super().__init__()
        self.decay = decay
        self.ema_model = None
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.backup_every_n_epochs = backup_every_n_epochs
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

    def on_fit_start(self, trainer, pl_module):
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(pl_module.model)
            self.ema_model.eval()
            self.ema_model.requires_grad_(False)
        self.ema_model.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), pl_module.model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1.0 - self.decay
                )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (self.backup_dir and epoch > 0
                and epoch % self.backup_every_n_epochs == 0):
            backup_path = self.backup_dir / f"ema_epoch_{epoch:04d}.pt"
            torch.save(self.ema_model.state_dict(), backup_path)
            print(f"💾 EMA backup → {backup_path}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['ema_state_dict'] = self.ema_model.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'ema_state_dict' in checkpoint:
            if self.ema_model is None:
                self.ema_model = copy.deepcopy(pl_module.model)
                self.ema_model.eval()
                self.ema_model.requires_grad_(False)
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.ema_model.to(pl_module.device)
            print("✅ EMA weights restored from checkpoint")


class GenerationVisualizationCallback(pl.Callback):
    def __init__(self, ema_callback, save_dir="ddpm_samples",
                 every_n_epochs=10, num_inference_steps=50, guidance_scale=2.0):
        super().__init__()
        self.ema_callback = ema_callback
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        ema_model = self.ema_callback.ema_model
        if ema_model is None:
            return
        ema_model.eval()
        scheduler = pl_module.scheduler
        device = pl_module.device
        null_class = pl_module.null_class
        num_classes = pl_module.hparams.num_classes
        images = []
        timesteps = scheduler.get_ddim_timesteps(self.num_inference_steps)
        for cls in range(num_classes):
            x = torch.randn(1, 3, 128, 128, device=device)
            class_label = torch.tensor([cls], device=device)
            null_label = torch.tensor([null_class], device=device)
            for i in range(len(timesteps)):
                t_curr = torch.tensor([timesteps[i]], device=device)
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                noise_cond = ema_model(x, t_curr, class_label)
                noise_uncond = ema_model(x, t_curr, null_label)
                noise_pred = noise_uncond + self.guidance_scale * (
                    noise_cond - noise_uncond
                )
                x = scheduler.ddim_reverse_step(
                    x, noise_pred, timesteps[i], t_prev
                )
            images.append(x.squeeze(0))
        grid = vutils.make_grid(
            torch.stack(images), nrow=num_classes, normalize=True
        )
        save_path = self.save_dir / f"epoch_{epoch:04d}.png"
        vutils.save_image(grid, save_path)
        print(f"🖼️  Sample grid → {save_path}")


class DDPMLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, num_timesteps=1000, num_classes=5,
                 cfg_drop_rate=0.15, max_epochs=300):
        super().__init__()
        self.save_hyperparameters()

        self.model = ClassConditionalUNet(
            c_in=3, c_out=3, base_channels=64, num_classes=num_classes
        )
        self.scheduler = CosineNoiseScheduler(num_timesteps=num_timesteps)
        self.null_class = num_classes

        # Per-class loss weights: inverse sqrt frequency
        self.register_buffer(
            "class_loss_weights",
            torch.tensor([1.0, 2.21, 1.35, 3.06, 2.48])
        )

    def training_step(self, batch, batch_idx):
        images, labels = batch

        t = torch.randint(
            0, self.scheduler.num_timesteps,
            (images.size(0),), device=self.device
        ).long()

        noise = torch.randn_like(images)
        noisy_images = self.scheduler.add_noise(images, noise, t)

        drop_mask = (
            torch.rand(labels.shape[0], device=self.device)
            < self.hparams.cfg_drop_rate
        )
        cfg_labels = labels.clone()
        cfg_labels[drop_mask] = self.null_class

        noise_pred = self.model(noisy_images, t, cfg_labels)

        # Per-sample MSE (unreduced)
        per_sample_loss = F.mse_loss(
            noise_pred, noise, reduction='none'
        ).mean(dim=[1, 2, 3])

        # Assign weight 1.0 to CFG dropped samples, otherwise use class weights
        sample_weights = torch.where(
            drop_mask,
            torch.ones_like(per_sample_loss),
            self.class_loss_weights[labels]
        )

        loss = (per_sample_loss * sample_weights).mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
