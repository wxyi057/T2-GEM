from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import wandb
from monai.data import Dataset
from monai.data.image_reader import NumpyReader
from monai.metrics import SSIMMetric
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.transforms import Compose, EnsureTyped, LoadImaged
from monai.utils import set_determinism

from t2gem.utils.device import get_amp_dtype_and_device, print_model_info
from t2gem.utils.logger import get_logger, init_wandb
from t2gem.lfm.ema import EMA
from t2gem.lfm.diffusion import ForwardDiffusion
from t2gem.lfm.flow import FlowModel
from t2gem.utils.optim import adjust_learning_rate

logger = get_logger(__name__)


def get_dataloader(config: DictConfig, stage: str = "train") -> DataLoader:
    df = pd.read_csv(config.data.csv_path)
    if stage not in {"train", "val"}:
        raise ValueError("stage must be 'train' or 'val'.")
    df = df[df["split"] == stage]

    cond_key = f"{config.data.cond}_latent"
    image_key = f"{config.data.image}_latent"
    records = df[[cond_key, image_key]].to_dict(orient="records")
    npz_reader = NumpyReader(npz_keys=["data"])
    transforms = Compose(
        [
            LoadImaged(keys=[cond_key, image_key], reader=npz_reader),
            EnsureTyped(keys=[cond_key, image_key], dtype=torch.float32),
        ]
    )

    dataset = Dataset(data=records, transform=transforms)
    batch_size = (
        config.train.batch_size
        if stage == "train"
        else config.train.val_batch_size
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(stage == "train"),
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=bool(config.data.num_workers > 0),
        drop_last=(stage == "train"),
    )
    logger.info(
        f"Loaded {len(dataset)} samples for split '{stage}' "
        f"with batch_size={batch_size} and num_workers={config.data.num_workers}."
    )
    return loader


class LatentCondVelocityUNet(nn.Module):
    def __init__(
        self,
        unet: DiffusionModelUNet,
    ) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is not None:
            x = torch.cat([x, context], dim=1)
        out = self.unet(x=x, timesteps=t)
        return out


class LFMTrainer:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        set_determinism(seed=config.seed)
        self.amp_dtype, self.device = get_amp_dtype_and_device()
        self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"

        self.cond_key = f"{config.data.cond}_latent"
        self.image_key = f"{config.data.image}_latent"

        self.train_loader = get_dataloader(config, stage="train")
        self.val_loader = get_dataloader(config, stage="val")

        self.scale_factor_cond = float(config.scale_factor.cond)
        self.scale_factor_image = float(config.scale_factor.image)

        unet_cfg = config.model.unet
        self.unet = DiffusionModelUNet(
            spatial_dims=unet_cfg.spatial_dims,
            in_channels=unet_cfg.in_channels,
            out_channels=unet_cfg.out_channels,
            num_res_blocks=unet_cfg.num_res_blocks,
            channels=unet_cfg.num_channels,
            attention_levels=unet_cfg.attention_levels,
            norm_num_groups=unet_cfg.norm_num_groups,
            num_head_channels=unet_cfg.num_head_channels,
            resblock_updown=unet_cfg.resblock_updown,
            upcast_attention=True,
        ).to(self.device)
        print_model_info(self.unet)
        self.net = LatentCondVelocityUNet(
            unet=self.unet
        )
        flow_cfg = config.flow
        self.flow = FlowModel(
            net_cfg=self.net,
            schedule=flow_cfg.schedule,
            sigma_min=flow_cfg.sigma_min,
        ).to(self.device)

        self.start_from_noise = flow_cfg.start_from_noise

        self.noising_step = flow_cfg.noising_step
        self.noise_image = self.noising_step > 0
        if self.start_from_noise and self.noise_image:
            raise ValueError("Cannot use noising step with start_from_noise=True")
        if self.noising_step > 0:
            self.diffusion = ForwardDiffusion()
        else:
            self.diffusion = None

        # ema
        ema_cfg = config.ema
        self.ema = EMA(
            self.flow,
            beta=ema_cfg.beta,
            update_after_step=ema_cfg.update_after_step,
            update_every=ema_cfg.update_every,
            power=3/4.,
            include_online_model=False
        ).to(self.device)
        self.use_ema_for_sampling = ema_cfg.use_ema_for_sampling

        # optim
        self.base_lr = float(config.optim.lr)
        self.min_lr = float(config.optim.min_lr)
        self.warmup_steps = int(config.optim.warmup_steps)
        self.weight_decay = float(config.optim.weight_decay)

        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        steps_per_epoch = max(len(self.train_loader), 1)
        self.max_epochs = int(config.train.epochs)
        self.max_steps = int(self.max_epochs * steps_per_epoch)
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"Invalid warmup schedule: warmup_steps={self.warmup_steps} must be < max_steps={self.max_steps} "
                f"(epochs={self.max_epochs}, steps/epoch={steps_per_epoch})"
            )

        self.use_amp = bool(config.train.use_amp)
        self.scaler = GradScaler(enabled=(self.use_amp and torch.cuda.is_available()))

        self.log_every = int(config.train.log_every)
        self.grad_clip_norm = config.train.grad_clip_norm
        self.val_every = int(config.train.val_every)
        self.save_every = int(config.train.save_every)

        tags = ["lfm", f"{config.data.cond}_to_{config.data.image}", f"seed{config.seed}"]
        self.wandb_run, self.ckpt_dir = init_wandb(config, tags=tags)

        self.autoencoder = self._init_autoencoder()

        self.val_metric_batches = int(config.train.val_num_batches)
        self.ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="mean")

        self.global_step = 0
        self.start_epoch = 0
        if config.train.resume:
            self.load_checkpoint(config.train.resume)

    def _init_autoencoder(self) -> AutoencoderKL | None:
        ae_cfg = self.config.autoencoder
        ckpt_value = ae_cfg.ckpt
        ckpt_path = str(ckpt_value).strip()

        autoencoder = AutoencoderKL(
            spatial_dims=ae_cfg.spatial_dims,
            in_channels=ae_cfg.in_channels,
            out_channels=ae_cfg.out_channels,
            num_res_blocks=ae_cfg.num_res_blocks,
            channels=ae_cfg.num_channels,
            attention_levels=ae_cfg.attention_levels,
            latent_channels=ae_cfg.z_channels,
            norm_num_groups=ae_cfg.norm_num_groups,
        ).to(self.device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"]
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()
        autoencoder.requires_grad_(False)
        logger.info(f"Loaded frozen autoencoder from {ckpt_path}")
        return autoencoder

    def train_one_epoch(self, epoch_index: int) -> float:
        self.flow.train()
        running_loss = 0.0

        for _, batch in enumerate(self.train_loader):
            lr = adjust_learning_rate(
                optimizer=self.optimizer,
                step=float(self.global_step + 1),
                warmup_steps=self.warmup_steps,
                max_n_steps=self.max_steps,
                lr=self.base_lr,
                min_lr=self.min_lr,
            )

            cond = batch[self.cond_key].to(self.device) * self.scale_factor_cond
            target = batch[self.image_key].to(self.device) * self.scale_factor_image

            source = torch.randn_like(target) if self.start_from_noise else cond
            if self.noise_image:
                source = self.diffusion.q_sample(x_start=source, t=self.noising_step)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.autocast_device_type, dtype=self.amp_dtype, enabled=self.scaler.is_enabled()):
                loss = self.flow.training_losses(x1=target, x0=source, context=cond)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            running_loss += float(loss.item())

            if self.wandb_run is not None:
                log_payload = {
                    "train/loss": float(loss.item()),
                    "train/epoch": int(epoch_index),
                }
                if (self.global_step % self.log_every) == 0:
                    with torch.no_grad():
                        log_payload.update(
                            {
                                "train/lr": float(lr),
                                "train/cond_mean": float(cond.mean().detach().cpu()),
                                "train/cond_std": float(cond.std().detach().cpu()),
                                "train/target_mean": float(target.mean().detach().cpu()),
                                "train/target_std": float(target.std().detach().cpu()),
                            }
                        )
                    log_payload["train/grad_norm"] = float(grad_norm.detach().cpu())
                self.wandb_run.log(log_payload, step=self.global_step)

            self.global_step += 1

        avg_loss = running_loss / max(len(self.train_loader), 1)
        logger.info(f"Epoch {epoch_index}: train_loss={avg_loss:.4e}")
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch_index: int) -> float:
        self.flow.eval()
        total_loss = 0.0
        metric_batches = max(int(self.val_metric_batches), 0)
        metric_latent_mse = 0.0
        metric_ssim = 0.0
        metric_count = 0

        sample_kwargs = None
        flow_for_sampling = None
        if metric_batches > 0:
            sample_kwargs = self._build_sample_kwargs()
            flow_for_sampling = self.flow
            if self.ema is not None and self.use_ema_for_sampling:
                flow_for_sampling = self.ema.ema_model
            flow_for_sampling.eval()

        for i, batch in enumerate(self.val_loader):
            cond_latent = batch[self.cond_key].to(self.device)
            target_latent = batch[self.image_key].to(self.device)
            cond = cond_latent * self.scale_factor_cond
            target = target_latent * self.scale_factor_image

            source = torch.randn_like(target) if self.start_from_noise else cond
            if self.noise_image:
                source = self.diffusion.q_sample(x_start=source, t=self.noising_step)
            with torch.autocast(self.autocast_device_type, dtype=self.amp_dtype, enabled=self.scaler.is_enabled()):
                loss = self.flow.training_losses(x1=target, x0=source, context=cond)
            total_loss += float(loss.item())

            if metric_count < metric_batches:
                init = source
                with torch.autocast(self.autocast_device_type, dtype=self.amp_dtype, enabled=self.scaler.is_enabled()):
                    pred_scaled = flow_for_sampling.generate(x=init, context=cond, sample_kwargs=sample_kwargs)
                latent_mse = F.mse_loss(pred_scaled.float(), target.float(), reduction="mean")
                metric_latent_mse += float(latent_mse.item())

                pred_latent = pred_scaled / self.scale_factor_image
                pred_img = self.autoencoder.decode(pred_latent)
                gt_img = self.autoencoder.decode(target_latent)

                pred_img_01 = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)
                gt_img_01 = torch.clamp((gt_img + 1.0) / 2.0, 0.0, 1.0)
                ssim_val = self.ssim_metric(pred_img_01.float(), gt_img_01.float())
                metric_ssim += float(ssim_val.mean().item())
                metric_count += 1

                if i == 0:
                    cond_img = self.autoencoder.decode(cond_latent)
                    self.visualization(
                        batch,
                        step=self.global_step,
                        pred_img=pred_img,
                        gt_img=gt_img,
                        cond_img=cond_img,
                    )
            elif i == 0:
                self.visualization(batch, step=self.global_step)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        log_payload = {"val/loss": avg_loss, "val/epoch": epoch_index}
        if metric_count > 0:
            log_payload["val/latent_mse_scaled"] = metric_latent_mse / metric_count
            log_payload["val/ssim"] = metric_ssim / metric_count
        if self.wandb_run is not None:
            self.wandb_run.log(log_payload, step=self.global_step)
        if metric_count > 0:
            logger.info(
                f"Epoch {epoch_index}: val_loss={avg_loss:.4e} "
                f"val_latent_mse_scaled={log_payload['val/latent_mse_scaled']:.4e} "
                f"val_ssim={log_payload['val/ssim']:.4f}"
            )
        else:
            logger.info(f"Epoch {epoch_index}: val_loss={avg_loss:.4e}")
        return avg_loss

    def _build_sample_kwargs(self) -> dict:
        cfg = self.config.sampling
        sample_kwargs = dict(
            num_steps=cfg.num_steps,
            method=cfg.method,
        )
        return sample_kwargs

    @torch.no_grad()
    def visualization(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        pred_scaled: torch.Tensor | None = None,
        pred_img: torch.Tensor | None = None,
        gt_img: torch.Tensor | None = None,
        cond_img: torch.Tensor | None = None,
    ) -> None:
        cond_latent = batch[self.cond_key].to(self.device)
        target_latent = batch[self.image_key].to(self.device)

        cond_scaled = cond_latent * self.scale_factor_cond
        target_scaled = target_latent * self.scale_factor_image

        if pred_img is None or gt_img is None:
            if pred_scaled is None:
                init = torch.randn_like(target_scaled) if self.start_from_noise else cond_scaled
                if self.noise_image:
                    init = self.diffusion.q_sample(x_start=init, t=self.noising_step)

                sample_kwargs = self._build_sample_kwargs()

                flow_for_sampling = self.flow
                if self.ema is not None and self.use_ema_for_sampling:
                    flow_for_sampling = self.ema.ema_model
                flow_for_sampling.eval()

                with torch.autocast(self.autocast_device_type, dtype=self.amp_dtype, enabled=self.scaler.is_enabled()):
                    pred_scaled = flow_for_sampling.generate(x=init, context=cond_scaled, sample_kwargs=sample_kwargs)
            pred_latent = pred_scaled / self.scale_factor_image
            if pred_img is None:
                pred_img = self.autoencoder.decode(pred_latent)
            if gt_img is None:
                gt_img = self.autoencoder.decode(target_latent)

        if cond_img is None:
            cond_img = self.autoencoder.decode(cond_latent)

        mid = int(pred_img.shape[2] // 2)
        cond_slice = cond_img[0, 0, mid].detach().to(torch.float32).cpu()
        gt_slice = gt_img[0, 0, mid].detach().to(torch.float32).cpu()
        pred_slice = pred_img[0, 0, mid].detach().to(torch.float32).cpu()

        def _to_uint8(x: torch.Tensor) -> np.ndarray:
            vis = (x + 1.0) / 2.0
            return (torch.clamp(vis, 0.0, 1.0) * 255.0).to(torch.uint8).numpy()

        log_images = {
            "viz/cond": wandb.Image(_to_uint8(cond_slice)),
            "viz/gt": wandb.Image(_to_uint8(gt_slice)),
            "viz/pred": wandb.Image(_to_uint8(pred_slice)),
        }
        self.wandb_run.log(log_images, step=step)

    def save_checkpoint(self, epoch: int) -> Path:
        ckpt_dir = Path(self.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"ckpt_{epoch}.pt"
        to_save = {
            "model": self.flow.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }
        if self.ema is not None:
            to_save["ema"] = self.ema.state_dict()
        torch.save(to_save, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | os.PathLike) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.flow.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])

        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        logger.info(f"Loaded checkpoint from {ckpt_path} (epoch={self.start_epoch}, step={self.global_step}).")

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.max_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs}")
            self.train_one_epoch(epoch)

            if ((epoch + 1) % self.val_every) == 0:
                self.validate(epoch)
            if ((epoch + 1) % self.save_every) == 0:
                self.save_checkpoint(epoch)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    trainer = LFMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
