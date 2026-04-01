import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from omegaconf import DictConfig

from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from t2gem.autoencoder.losses import PerceptualWithDiscriminator
from t2gem.autoencoder.prostate import ProstateDataset
from t2gem.utils.device import (
    get_amp_dtype_and_device,
    print_model_info,
)
from t2gem.utils.logger import get_logger, init_wandb
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler
import wandb
import hydra
logger = get_logger(__name__)

class AutoencoderKLTrainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.image_types = list(config.data.image_types)
        self.image_key = "image"
        set_determinism(seed=config.seed)

        self.amp_dtype, self.device = get_amp_dtype_and_device()

        self.autoencoderkl = AutoencoderKL(
            spatial_dims=config.model.spatial_dims,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            num_res_blocks=config.model.num_res_blocks,
            channels=config.model.num_channels,
            attention_levels=config.model.attention_levels,
            latent_channels=config.model.z_channels,
            norm_num_groups=config.model.norm_num_groups,
        ).to(self.device)

        self.loss = PerceptualWithDiscriminator(
            kl_weight=config.loss.kl_weight,
            perceptual_weight=config.loss.perceptual_weight,
            disc_weight=config.loss.disc_weight,
            disc_start=config.loss.disc_start,
            spatial_dims=config.model.spatial_dims,
        ).to(self.device)

        self.discriminator = PatchDiscriminator(
            spatial_dims=config.model.spatial_dims,
            num_layers_d=config.loss.disc_num_layers,
            channels=config.loss.disc_num_channels,
            in_channels=config.loss.disc_in_channels,
            out_channels=config.loss.disc_out_channels,
            norm=config.loss.disc_norm,
        ).to(self.device)

        lr_g = config.optim.lr_g
        lr_d = config.optim.lr_d

        self.opt_g = torch.optim.Adam(self.autoencoderkl.parameters(), lr=lr_g)
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)

        self.use_amp = config.train.use_amp
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()

        root_dir = config.data.root
        image_size = config.data.image_size
        num_workers = config.data.num_workers

        self.train_dataset = ProstateDataset(
            root_dir=root_dir,
            split="train",
            image_types=self.image_types,
            image_size=image_size,
        )
        self.val_dataset = ProstateDataset(
            root_dir=root_dir,
            split="val",
            image_types=self.image_types,
            image_size=image_size,
        )

        bs = config.train.batch_size
        train_sampler = RandomSampler(self.train_dataset)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=bs,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_sampler = SequentialSampler(self.val_dataset)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.max_epochs = config.train.epochs

        image_tag = "+".join(self.image_types)
        tags=["autoencoderkl", image_tag, f"seed{config.seed}"]
        self.wandb_run, self.ckpt_dir = init_wandb(config, tags=tags)
        self.val_every = config.train.val_every
        self.save_every = config.train.save_every

        self.start_epoch = 0
        self.global_step = 0

        resume_path = config.train.resume
        if resume_path:
            self.load_checkpoint(resume_path)

        print_model_info(self.autoencoderkl)

    def train_one_epoch(self, epoch: int) -> dict:
        self.autoencoderkl.train()
        self.discriminator.train()

        for it, batch in enumerate(self.train_loader):
            inputs = batch[self.image_key].to(self.device)
            self.opt_g.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=(self.use_amp and torch.cuda.is_available())):
                reconstructions, z_mu, z_sigma = self.autoencoderkl(inputs)
                logits_fake = self.discriminator(reconstructions.contiguous().float())
                loss_g, log_g = self.loss(
                    inputs, 
                    reconstructions, 
                    z_mu, 
                    z_sigma, 
                    optimizer_idx=0, 
                    global_step=self.global_step, 
                    logits_fake=logits_fake, 
                    split="train"
                )

            self.scaler_g.scale(loss_g).backward()
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()
            self.wandb_run.log(log_g, step=self.global_step)
            self.opt_d.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=(self.use_amp and torch.cuda.is_available())):
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                logits_real = self.discriminator(inputs.contiguous().detach())
                loss_d, log_d = self.loss(
                    inputs, 
                    reconstructions, 
                    z_mu, 
                    z_sigma, 
                    optimizer_idx=1, 
                    global_step=self.global_step, 
                    logits_fake=logits_fake, 
                    logits_real=logits_real, 
                    split="train"
                )

            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()
            self.wandb_run.log(log_d, step=self.global_step)
            self.global_step += 1
        return {"epoch": epoch, "global_step": self.global_step}

    @torch.no_grad()
    def validate(self) -> dict:
        self.autoencoderkl.eval()
        self.discriminator.eval()

        totals = {}
        n_batches = 0
        image_previews: dict[str, dict[str, torch.Tensor]] = {}
        for batch in self.val_loader:
            inputs = batch[self.image_key].to(self.device)
            image_type = batch.get("image_type", None)
            if isinstance(image_type, (list, tuple)):
                image_type = image_type[0]
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=(self.use_amp and torch.cuda.is_available())):
                reconstructions, z_mu, z_sigma = self.autoencoderkl(inputs)
                logits_fake = self.discriminator(reconstructions.contiguous().float())
                loss_g, log_g = self.loss(
                    inputs, 
                    reconstructions, 
                    z_mu, 
                    z_sigma, 
                    optimizer_idx=0, 
                    global_step=self.global_step, 
                    logits_fake=logits_fake, 
                    split="val")
            for k, v in log_g.items():
                value = v.item() if isinstance(v, torch.Tensor) else v
                totals[k] = totals.get(k, 0.0) + float(value)
            n_batches += 1
            if image_type and image_type not in image_previews:
                mid_slice = inputs.shape[2] // 2
                inp_t = inputs[0, 0, mid_slice].detach().to(torch.float32).cpu()
                rec_t = reconstructions[0, 0, mid_slice].detach().to(torch.float32).cpu()
                image_previews[image_type] = {
                    "input": inp_t,
                    "reconstruction": rec_t,
                }

        if n_batches > 0:
            for k in list(totals.keys()):
                totals[k] = totals[k] / n_batches
            log_images = None
            if image_previews:
                log_images = {}
                for img_type, tensors in image_previews.items():
                    inp_vis = (tensors["input"] + 1) / 2
                    rec_vis = (tensors["reconstruction"] + 1) / 2
                    inp_np = (torch.clamp(inp_vis, 0.0, 1.0) * 255.0).to(torch.uint8).numpy()
                    rec_np = (torch.clamp(rec_vis, 0.0, 1.0) * 255.0).to(torch.uint8).numpy()
                    log_images[f"images/{img_type}/inputs"] = wandb.Image(inp_np)
                    log_images[f"images/{img_type}/reconstructions"] = wandb.Image(rec_np)
        else:
            log_images = None

        if totals:
            self.wandb_run.log(totals, step=self.global_step)
            if log_images is not None:
                self.wandb_run.log(log_images, step=self.global_step)
        val_metrics_str = {k: v if isinstance(v, int) else f"{v:.2e}" for k, v in totals.items()}
        logger.info(f"Validation metrics: {val_metrics_str}.")

        return totals

    def train(self) -> None:

        for epoch in range(self.start_epoch, self.max_epochs):
            logger.info(f"Epoch {epoch+1}/{self.max_epochs}")
            self.train_one_epoch(epoch)

            if ((epoch + 1) % self.val_every) == 0:
                val_metric = self.validate()
            if ((epoch + 1) % self.save_every) == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int) -> Path:
        ckpt_dir = Path(self.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"ckpt_{epoch}.pt"
        to_save = {
            "model": self.autoencoderkl.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d": self.scaler_d.state_dict(),
            "epoch": epoch + 1,
            "global_step": self.global_step,
        }
        torch.save(to_save, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | os.PathLike) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.autoencoderkl.load_state_dict(ckpt["model"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.scaler_g.load_state_dict(ckpt["scaler_g"])
        self.scaler_d.load_state_dict(ckpt["scaler_d"])
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        logger.info(f"Loaded checkpoint from {ckpt_path} (epoch={self.start_epoch}, step={self.global_step}).")

@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    trainer = AutoencoderKLTrainer(config)
    trainer.train() 

if __name__ == "__main__":
    main()
