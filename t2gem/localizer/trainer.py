"""Trainer for the localizer classification task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.nn import functional as F  # noqa: N812
from monai.losses import FocalLoss
import hydra

from t2gem.localizer.prostate import ProstateDataset
from t2gem.localizer.utils import (
    classification_metrics,
    emd_loss,
    get_classification_model,
    resolve_model_spec,
    get_weights,
)
from t2gem.utils.optim import EarlyStopping, GradScaler, adjust_learning_rate, get_n_accum_steps
from t2gem.utils.device import get_amp_dtype_and_device, print_model_info
from t2gem.utils.logger import get_logger, init_wandb
from monai.utils import set_determinism

logger = get_logger(__name__)


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: DictConfig,
) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError(f"logits must have at least 2 dims, got shape {tuple(logits.shape)}.")
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match logits shape excluding classes {tuple(logits.shape[:-1])}."
        )

    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.long().reshape(-1)
    weights = get_weights(config).to(logits.device)
    return emd_loss(
        logits=logits_flat,
        targets=labels_flat,
        num_classes=config.backbone.roi_classes,
        r=config.backbone.emd_r,
        weights=weights,
    )

class Trainer:
    """Localizer trainer following a simplified, self-contained loop."""

    def __init__(self, config: DictConfig):
        self.config = config
        resolve_model_spec(config)
        self.amp_dtype, self.device = get_amp_dtype_and_device()
        self.use_amp = config.use_amp
        self.scaler = GradScaler()

        set_determinism(config.seed)
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders(config)

        self.model = get_classification_model(config).to(self.device)
        print_model_info(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.backbone.lr,
            betas=config.backbone.betas,
            weight_decay=config.backbone.weight_decay
        )
        if self.scheduler_name == "cosine":
            warmup_epochs = config.backbone.n_warmup_epochs
            max_epochs = config.backbone.n_epochs
            if max_epochs <= 0:
                raise ValueError(f"backbone.n_epochs must be > 0, got {max_epochs}.")
            if warmup_epochs < 0:
                raise ValueError(f"backbone.n_warmup_epochs must be >= 0, got {warmup_epochs}.")
            if warmup_epochs >= max_epochs:
                raise ValueError(
                    f"backbone.n_warmup_epochs must be < backbone.n_epochs, got {warmup_epochs} >= {max_epochs}."
                )
            logger.info(
                "Enabled warmup+cosine LR schedule (epoch-based): warmup=%s/%s epochs, base_lr=%s, min_lr=%s",
                warmup_epochs,
                max_epochs,
                float(config.backbone.lr),
                float(getattr(config.backbone, "min_lr", 0.0)),
            )

        self.n_accum_steps = get_n_accum_steps(
            batch_size=config.backbone.batch_size,
            batch_size_per_device=config.backbone.batch_size_per_device,
            world_size=1,
        )
        self.clip_grad_norm = config.backbone.grad_clip_norm

        tags = [
            config.data.name,
            config.backbone.name,
            f"seed{config.seed}",
            f"modalities_{'-'.join(config.data.modalities)}",
            "isup4",
        ]
        self.wandb_run, self.ckpt_dir = init_wandb(config=config, tags=sorted(set(tags)))

        self.val_every = config.backbone.eval_interval
        self.save_every = config.backbone.save_checkpoint_interval
        self.early_stop_mode = config.backbone.early_stopping.mode
        self.early_stop = EarlyStopping(
            min_delta=config.backbone.early_stopping.min_delta,
            patience=config.backbone.early_stopping.patience,
        )
        self.best_metric = -float("inf") if self.early_stop_mode == "max" else float("inf")
        self.best_ckpt: Path | None = None
        self.start_epoch = 0
        self.global_step = 0

        resume_path = config.resume
        if resume_path:
            self.load_checkpoint(resume_path)

    def _build_dataloaders(
        self, config: DictConfig
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        data_dir = Path(config.data.root_dir).expanduser()
        metadata_name = config.data.metadata_name
        meta_df = pd.read_csv(data_dir / metadata_name, dtype={"case_id": str})

        train_meta = meta_df[meta_df["split"] == "train"].reset_index(drop=True)
        val_meta = meta_df[meta_df["split"] == "val"].reset_index(drop=True)
        test_meta = meta_df[meta_df["split"] == "test"].reset_index(drop=True)

        train_ds = ProstateDataset(config=config, meta_df=train_meta, is_train=True)
        val_ds = ProstateDataset(config=config, meta_df=val_meta, is_train=False)
        test_ds = ProstateDataset(config=config, meta_df=test_meta, is_train=False)
        
        train_sampler = RandomSampler(train_ds)
        train_loader = DataLoader(
            dataset=train_ds,
            sampler=train_sampler,
            batch_size=config.backbone.batch_size_per_device,
            drop_last=True,
            pin_memory=True,
            num_workers=config.backbone.num_workers_per_device,
        )

        def _make_eval_loader(ds: ProstateDataset) -> DataLoader:
            return DataLoader(
                dataset=ds,
                sampler=SequentialSampler(ds),
                batch_size=1,
                drop_last=False,
                pin_memory=True,
                num_workers=config.backbone.num_workers_per_device,
            )

        return train_loader, _make_eval_loader(val_ds), _make_eval_loader(test_ds)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss: List[float] = []
        self.optimizer.zero_grad(set_to_none=True)
        clip_grad = self.clip_grad_norm if self.clip_grad_norm > 0 else None

        for step, batch in enumerate(self.train_loader):
            lr = None
            if self.scheduler_name == "cosine":
                steps_per_epoch = max(len(self.train_loader), 1)
                lr = adjust_learning_rate(
                    optimizer=self.optimizer,
                    step=step / steps_per_epoch + epoch,
                    warmup_steps=int(self.config.backbone.n_warmup_epochs),
                    max_n_steps=int(self.config.backbone.n_epochs),
                    lr=float(self.config.backbone.lr),
                    min_lr=float(self.config.backbone.min_lr),
                )

            t2 = batch["t2"].to(self.device)
            rois = batch["rois"].to(self.device)
            labels = batch["zonal_labels"].to(self.device)

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(t2, rois)
                full_loss = _classification_loss(logits, labels, self.config)

            loss = full_loss / self.n_accum_steps
            update_grad = (step + 1) % self.n_accum_steps == 0

            grad_norm = self.scaler(
                loss=loss,
                optimizer=self.optimizer,
                clip_grad=clip_grad,
                parameters=self.model.parameters(),
                update_grad=update_grad,
            )

            if update_grad:
                self.optimizer.zero_grad(set_to_none=True)

            running_loss.append(full_loss.item())
            self.global_step += 1

            if self.wandb_run is not None:
                log_dict = {"train_loss": full_loss.item(), "epoch": epoch}
                if lr is not None:
                    log_dict["lr"] = float(lr)
                if grad_norm is not None:
                    try:
                        log_dict["grad_norm"] = float(grad_norm.item())
                    except Exception:  
                        pass
                self.wandb_run.log(log_dict, step=self.global_step)

        mean_loss = float(np.mean(running_loss)) if running_loss else 0.0
        lr_now = float(self.optimizer.param_groups[0].get("lr", self.config.backbone.lr))
        logger.info("Epoch %s | train_loss %.4f | lr %.6g", epoch, mean_loss, lr_now)
        return {"train_loss": mean_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        return self._evaluate_loader(self.val_loader, split="val")

    @torch.no_grad()
    def evaluation(self, ckpt_path: Path | None = None) -> Dict[str, float]:
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        val_labels, val_probs, _, _, val_zone_regions, val_zone_region_weights = self._collect_outputs(self.val_loader)
        val_metrics = self._metrics_from_arrays(
            labels_np=val_labels,
            probs_np=val_probs,
            zone_regions_np=val_zone_regions,
            zone_region_weights_np=val_zone_region_weights,
            split="val",
        )
        if self.wandb_run is not None:
            log_dict = dict(val_metrics)
            self.wandb_run.log(log_dict, step=self.global_step)
        logger.info("Val (fixed 0.5 thresholds) metrics: %s", {k: f"{v:.4f}" for k, v in val_metrics.items()})

        test_metrics = self._evaluate_loader(
            self.test_loader,
            split="test",
            save_outputs=True,
        )

        self._save_test_results_csv({
            "val": val_metrics,
            "promis_test": test_metrics,
        })
        return test_metrics

    def _collect_outputs(
        self, loader: DataLoader
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray | None, np.ndarray | None
    ]:
        self.model.eval()
        true_labels: List[torch.Tensor] = []
        pred_logits: List[torch.Tensor] = []
        zone_regions: List[torch.Tensor] = []
        zone_region_weights: List[torch.Tensor] = []
        case_ids: List[str] = []

        for batch in loader:
            t2 = batch["t2"].to(self.device)
            rois = batch["rois"].to(self.device)
            labels = batch["zonal_labels"].to(self.device)
            patient_id = batch.get("patient_id", None)
            zr = batch.get("zone_regions", None)
            zrw = batch.get("zone_region_weights", None)

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(t2, rois)

            true_labels.append(labels.cpu().to(torch.int64))
            pred_logits.append(logits.cpu().to(torch.float32))
            if zr is not None:
                zone_regions.append(zr.cpu().to(torch.int64))
            if zrw is not None:
                zone_region_weights.append(zrw.cpu().to(torch.float32))

            if patient_id is not None:
                pid_list = patient_id if isinstance(patient_id, (list, tuple)) else [patient_id]
                if len(pid_list) != labels.shape[0]:
                    pid_list = [pid_list[0]] * labels.shape[0]
                case_ids.extend([str(pid) for pid in pid_list])

        if not pred_logits:
            return (
                np.zeros((0, 20), dtype=np.int64),
                np.zeros((0, 20, 0), dtype=np.float32),
                np.zeros((0, 20)),
                [],
                None,
                None,
            )

        labels_np = torch.cat(true_labels, dim=0).detach().numpy()
        logits_tensor = torch.cat(pred_logits, dim=0).detach()
        probs_np = F.softmax(logits_tensor, dim=-1).detach().numpy()
        pred_labels = np.argmax(probs_np, axis=-1).astype(np.float32)

        zone_regions_np: np.ndarray | None = None
        if zone_regions:
            zone_regions_np = torch.cat(zone_regions, dim=0).detach().numpy()

        zone_region_weights_np: np.ndarray | None = None
        if zone_region_weights:
            zone_region_weights_np = torch.cat(zone_region_weights, dim=0).detach().numpy()

        if zone_regions_np is None and zone_region_weights_np is None:
            raise ValueError(
                "Missing both 'zone_regions' and 'zone_region_weights' in dataloader batch. "
                "Please ensure the Dataset provides per-case PZ/TZ mapping for val/test."
            )

        return labels_np, probs_np, pred_labels, case_ids, zone_regions_np, zone_region_weights_np

    def _metrics_from_arrays(
        self,
        labels_np: np.ndarray,
        probs_np: np.ndarray,
        zone_regions_np: np.ndarray | None,
        split: str,
        zone_region_weights_np: np.ndarray | None = None,
    ) -> Dict[str, float]:
        raw_metrics = classification_metrics(
            true_labels=labels_np,
            pred_probs=probs_np,
            zone_regions=zone_regions_np,
            zone_region_weights=zone_region_weights_np,
        )

        metrics: Dict[str, float] = {}
        for level, level_metrics in raw_metrics.items():
            prefix = f"{level}_{split}"
            for name, val in level_metrics.items():
                metrics[f"{prefix}/{name}"] = float(val)
        return metrics

    def _evaluate_loader(
        self,
        loader: DataLoader,
        split: str,
        save_outputs: bool = False,
    ) -> Dict[str, float]:
        labels_np, probs_np, pred_labels, case_ids, zone_regions_np, zone_region_weights_np = self._collect_outputs(loader)
        if probs_np.shape[0] == 0:
            logger.warning("No samples found for split=%s.", split)
            return {}
        metrics = self._metrics_from_arrays(
            labels_np=labels_np,
            probs_np=probs_np,
            zone_regions_np=zone_regions_np,
            zone_region_weights_np=zone_region_weights_np,
            split=split,
        )

        if self.wandb_run is not None:
            log_dict = dict(metrics)
            self.wandb_run.log(log_dict, step=self.global_step)

        logger.info("%s metrics: %s", split.capitalize(), {k: f"{v:.4f}" for k, v in metrics.items()})

        if save_outputs:
            self._save_predictions(case_ids, labels_np, pred_labels, probs_np, split)
        return metrics

    def _get_monitor_metric(self, metrics: Dict[str, float], metric_name: str, split: str) -> float | None:
        candidates = [
            metric_name,
            f"{split}_{metric_name}",
            f"{metric_name}_{split}",
            f"{split}/{metric_name}",
        ]
        for key in candidates:
            if key in metrics:
                return metrics[key]
        return None

    def _save_predictions(
        self,
        case_ids: List[str],
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        pred_probs: np.ndarray,
        split: str,
    ) -> None:
        if self.ckpt_dir is None:
            return
        out_dir = Path(self.ckpt_dir) / f"{split}_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "case_id": case_ids[: true_labels.shape[0]],
                "true_label": true_labels.tolist(),
                "pred_label": pred_labels.tolist(),
                "pred_probability": pred_probs.tolist(),
            }
        )
        df.to_csv(out_dir / "classification_prediction.csv", index=False)
        logger.info("Saved %s predictions to %s", split, out_dir)

    def _save_test_results_csv(
        self,
        metrics_by_source: Dict[str, Dict[str, float]],
    ) -> None:
        """Save all final metrics to CSV and JSON."""
        if not self.ckpt_dir:
            return
        ckpt_dir = Path(self.ckpt_dir)

        # ── CSV (long format) ──
        out_csv = ckpt_dir / "test_results.csv"
        rows = []
        for source, source_metrics in metrics_by_source.items():
            for key, val in source_metrics.items():
                rows.append({"source": source, "metric": key, "value": val})
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        logger.info("Saved all test results CSV to %s", out_csv)

        # ── JSON (nested, easy to load programmatically) ──
        out_json = ckpt_dir / "final_metrics.json"
        json_dict = metrics_by_source
        with open(out_json, "w") as f:
            json.dump(json_dict, f, indent=2)
        logger.info("Saved all final metrics JSON to %s", out_json)

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.config.backbone.n_epochs):
            self.train_one_epoch(epoch)

            do_val = (epoch + 1) % self.val_every == 0
            if do_val:
                val_metrics = self.validate()
                monitor = self._get_monitor_metric(
                    metrics=val_metrics,
                    metric_name=self.config.backbone.early_stopping.metric,
                    split="val",
                )
                if monitor is not None:
                    metric_for_es = -monitor if self.early_stop_mode == "max" else monitor
                    self.early_stop.update(metric_for_es)
                    if self.early_stop.has_improved:
                        self.best_metric = monitor
                        self.best_ckpt = self.save_checkpoint(epoch, is_best=True)
                    else:
                        logger.info(
                            "No improvement. Patience %s/%s",
                            self.early_stop.patience_count,
                            self.early_stop.patience,
                        )
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch, is_best=False)

            if self.early_stop.should_stop:
                logger.info(
                    "Early stopping triggered at epoch %s with best metric %.4f.",
                    epoch,
                    self.best_metric,
                )
                break

        ckpt_for_eval = self.best_ckpt
        if ckpt_for_eval is None and self.ckpt_dir:
            ckpt_for_eval = Path(self.ckpt_dir) / "last.pt"
        self.evaluation(ckpt_for_eval)

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        ckpt_dir = Path(self.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        name = "best.pt" if is_best else "last.pt"
        ckpt_path = ckpt_dir / name
        to_save = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }
        torch.save(to_save, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_metric = float(ckpt.get("best_metric", self.best_metric))
        if hasattr(self, "early_stop"):
            self.early_stop.best_metric = -self.best_metric if self.early_stop_mode == "max" else self.best_metric
        logger.info(
            "Loaded checkpoint from %s (epoch=%s, step=%s, best_metric=%.4f).",
            ckpt_path,
            self.start_epoch,
            self.global_step,
            self.best_metric,
        )


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
