"""Classification model training and evaluation utilities."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    cohen_kappa_score,
)
from torch.nn import functional as F  # noqa: N812

from t2gem.utils.logger import get_logger
from t2gem.localizer.resnet import RoINet

from omegaconf import DictConfig
from torch import nn

logger = get_logger(__name__)


def emd_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    r: int = 2,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted Earth Mover's Distance loss."""
    if logits.shape[-1] != num_classes:
        raise ValueError(f"Expected logits last dim {num_classes}, got {logits.shape[-1]}.")
    
    probs = F.softmax(logits, dim=-1)
    targets_long = targets.to(torch.int64)
    one_hot = F.one_hot(targets_long, num_classes=num_classes).to(dtype=probs.dtype)
    
    cdf_pred = torch.cumsum(probs, dim=-1)
    cdf_true = torch.cumsum(one_hot, dim=-1)
    
    dist = torch.pow(torch.abs(cdf_pred - cdf_true), r).mean(dim=-1)
    
    if weights is not None:
        sample_weights = weights[targets_long]
        return (dist * sample_weights).sum() / (sample_weights.sum() + 1e-8)
    
    return dist.mean()


def normalize_modalities(config: DictConfig) -> list[str]:
    """Parse and validate modalities; ensures deterministic ordering and de-dup."""
    raw = getattr(config.data, "modalities", ["t2"])
    modalities = [raw] if isinstance(raw, str) else list(raw)
    if not modalities:
        raise ValueError("config.data.modalities must contain at least one modality.")
    normalized: list[str] = []
    for mod in modalities:
        mod_norm = str(mod).lower()
        if mod_norm not in normalized:
            normalized.append(mod_norm)
    config.data.modalities = normalized
    return normalized


def resolve_model_spec(config: DictConfig) -> tuple[int, int]:
    modalities = normalize_modalities(config)
    in_channels = len(modalities)
    roi_classes = 4

    config.backbone.in_channels = in_channels
    config.backbone.roi_classes = roi_classes
    return in_channels, roi_classes


def get_classification_model(config: DictConfig) -> nn.Module:
    in_channels, roi_classes = resolve_model_spec(config)

    if config.backbone.name != "resnet":
        raise ValueError(f"Invalid model name {config.backbone.name}. Expected 'resnet'.")

    model = RoINet(
        model_depth=config.backbone.depth,
        in_channels=in_channels,
        roi_size=(7, 7, 7),
        roi_classes=roi_classes,
        sampling_ratio=config.backbone.sampling_ratio,
        use_crf=config.backbone.use_crf,
        canonical_scale=config.backbone.canonical_scale,
        canonical_level=config.backbone.canonical_level,
        crf_n_iter=config.backbone.crf_n_iter,
        crf_sigma_init=config.backbone.crf_sigma,
        crf_smoothness_init=config.backbone.crf_smoothness_init,
        crf_knn_k=config.backbone.crf_knn_k,
    )
    return model


def get_weights(config: DictConfig) -> torch.Tensor:
    """Return class weights for the fixed 4-class isup4 setup."""
    explicit = config.backbone.class_weights
    n_classes = config.backbone.roi_classes
    if n_classes != 4:
        raise ValueError(f"Localizer is fixed to 4 classes (isup4), got roi_classes={n_classes}.")

    if explicit is not None:
        weights = torch.as_tensor(explicit, dtype=torch.float32)
        if weights.numel() != n_classes:
            raise ValueError(
                f"class_weights length {weights.numel()} does not match roi_classes {n_classes}."
            )
        return weights

    return torch.ones(n_classes, dtype=torch.float32)


def _binary_operating_points(true_bin: np.ndarray, pos_scores: np.ndarray) -> dict[str, float]:
    """Compute Spec/Sens at fixed operating points with linear interpolation."""
    results = {
        "spec_at_sens80": np.nan,
        "spec_at_sens90": np.nan,
        "spec_at_sens95": np.nan,
        "sens_at_spec80": np.nan,
        "sens_at_spec90": np.nan,
        "sens_at_spec95": np.nan,
    }
    uniq = np.unique(true_bin)
    if uniq.size < 2:
        return results

    fpr, tpr, _ = roc_curve(true_bin, pos_scores, drop_intermediate=False)
    spec = 1.0 - fpr

    def _dedup_x_take_max_y(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return strictly increasing xs with ys aggregated by max over duplicate xs."""
        xs_arr = np.asarray(xs, dtype=np.float64).reshape(-1)
        ys_arr = np.asarray(ys, dtype=np.float64).reshape(-1)
        if xs_arr.size == 0:
            return xs_arr, ys_arr
        order = np.argsort(xs_arr, kind="mergesort")
        xs_sorted = xs_arr[order]
        ys_sorted = ys_arr[order]
        uniq_xs, idx_start = np.unique(xs_sorted, return_index=True)
        ys_max = np.maximum.reduceat(ys_sorted, idx_start)
        return uniq_xs, ys_max

    def _interp_at_x(x: float, xs: np.ndarray, ys: np.ndarray, fill: float = np.nan) -> float:
        """Linear interpolate y at x given monotonic xs."""
        xs_u, ys_u = _dedup_x_take_max_y(xs, ys)
        if xs_u.size == 0:
            return fill
        if x <= xs_u[0]:
            return float(ys_u[0])
        if x >= xs_u[-1]:
            return float(ys_u[-1])
        return float(np.interp(x, xs_u, ys_u))

    fpr_u, tpr_u = _dedup_x_take_max_y(fpr, tpr)
    tpr_u = np.maximum.accumulate(tpr_u)
    results["sens_at_spec80"] = _interp_at_x(0.20, fpr_u, tpr_u)
    results["sens_at_spec90"] = _interp_at_x(0.10, fpr_u, tpr_u)
    results["sens_at_spec95"] = _interp_at_x(0.05, fpr_u, tpr_u)

    tpr_u2, spec_u = _dedup_x_take_max_y(tpr, spec)
    spec_u = np.maximum.accumulate(spec_u[::-1])[::-1]
    results["spec_at_sens80"] = _interp_at_x(0.80, tpr_u2, spec_u)
    results["spec_at_sens90"] = _interp_at_x(0.90, tpr_u2, spec_u)
    results["spec_at_sens95"] = _interp_at_x(0.95, tpr_u2, spec_u)
    return results


def _coerce_hard_zone_regions(
    *,
    n_cases: int,
    n_zones: int,
    zone_regions: np.ndarray | None,
    zone_region_weights: np.ndarray | None,
) -> np.ndarray:
    """Return hard per-zone region ids with shape (n_cases, n_zones), values {1 (PZ), 2 (TZ)}."""
    if zone_regions is not None:
        zr = np.asarray(zone_regions, dtype=np.int64)
        if zr.shape != (n_cases, n_zones):
            raise ValueError(
                f"zone_regions must have shape (n_cases, n_zones)=({n_cases}, {n_zones}), got {zr.shape}."
            )
        if not np.isin(zr, [1, 2]).all():
            bad = np.unique(zr[~np.isin(zr, [1, 2])]).tolist()
            raise ValueError(f"zone_regions contains invalid values {bad}. Expected only {{1 (pz), 2 (tz)}}.")
        return zr

    if zone_region_weights is None:
        raise ValueError(
            "Need zone_regions (preferred for hard assignment) or zone_region_weights to derive hard PZ/TZ mapping. "
            "Expected zone_regions shape [n_cases, 20] with values pz=1, tz=2, "
            "or zone_region_weights shape [n_cases, 20, 2] with columns [pz, tz]."
        )

    w = np.asarray(zone_region_weights, dtype=np.float32)
    if w.shape != (n_cases, n_zones, 2):
        raise ValueError(
            f"zone_region_weights must have shape (n_cases, n_zones, 2)=({n_cases}, {n_zones}, 2), got {w.shape}."
        )
    if not np.isfinite(w).all():
        raise ValueError("zone_region_weights contains NaN/Inf values.")
    w = np.clip(w, 0.0, None)
    # np.argmax tie-breaks to the first column, i.e. PZ.
    return (np.argmax(w, axis=-1).astype(np.int64) + 1)


def _has_both_regions(case_hard_regions: np.ndarray) -> bool:
    """Return True when a case contains at least one PZ zone and one TZ zone."""
    r = np.asarray(case_hard_regions, dtype=np.int64).reshape(-1)
    return bool(np.any(r == 1) and np.any(r == 2))


def _region_mask(case_hard_regions: np.ndarray, region_key: str) -> np.ndarray:
    """Return a boolean mask for region_key in {'pz', 'tz'} on a hard-assigned case map."""
    if region_key == "pz":
        region_val = 1
    elif region_key == "tz":
        region_val = 2
    else:
        raise ValueError(f"region_key must be 'pz' or 'tz', got '{region_key}'.")
    r = np.asarray(case_hard_regions, dtype=np.int64).reshape(-1)
    return r == region_val


def _isup4_zone_metrics(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    zone_regions: np.ndarray | None = None,
    zone_region_weights: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Compute isup4 metrics at zone/patient/pz/tz levels."""
    n_classes = pred_probs.shape[-1]
    n_cases, n_zones = true_labels.shape
    true_flat = true_labels.reshape(-1)
    probs_flat = pred_probs.reshape(-1, n_classes)
    pred_flat = np.argmax(probs_flat, axis=1)
    fixed_threshold = 0.5

    metrics_zone: dict[str, float] = {}
    labels = list(range(n_classes))

    metrics_zone["f1_macro"] = f1_score(true_flat, pred_flat, average="macro", labels=labels)
    metrics_zone["f1_weighted"] = f1_score(true_flat, pred_flat, average="weighted", labels=labels)
    metrics_zone["accuracy"] = accuracy_score(true_flat, pred_flat)
    if np.unique(true_flat).size > 1:
        metrics_zone["mcc"] = matthews_corrcoef(true_flat, pred_flat)
        metrics_zone["auc_macro_ovr"] = roc_auc_score(
            y_true=true_flat,
            y_score=probs_flat,
            average="macro",
            multi_class="ovr",
            labels=labels,
        )
        metrics_zone["qwk"] = cohen_kappa_score(true_flat, pred_flat, weights="quadratic")
    else:
        metrics_zone["mcc"] = np.nan
        metrics_zone["auc_macro_ovr"] = np.nan
        metrics_zone["qwk"] = np.nan

    def _binary_block(
        prefix: str,
        true_bin: np.ndarray,
        pos_scores: np.ndarray,
    ) -> dict[str, float]:
        ops = _binary_operating_points(true_bin, pos_scores)
        out: dict[str, float] = {f"{prefix}_{k}": v for k, v in ops.items()}
        out[f"{prefix}_threshold"] = fixed_threshold

        pred_bin = (pos_scores >= fixed_threshold).astype(int)
        out[f"{prefix}_f1"] = f1_score(true_bin, pred_bin, zero_division=0)
        out[f"{prefix}_acc"] = accuracy_score(true_bin, pred_bin)
        if np.unique(true_bin).size > 1:
            out[f"{prefix}_auc"] = roc_auc_score(true_bin, pos_scores)
            out[f"{prefix}_mcc"] = matthews_corrcoef(true_bin, pred_bin)
        else:
            out[f"{prefix}_auc"] = np.nan
            out[f"{prefix}_mcc"] = np.nan
        return out

    thresholds_zone = {
        "gt0": (true_flat > 0, probs_flat[:, 1:].sum(axis=1)),
        "gt1": (true_flat > 1, probs_flat[:, 2:].sum(axis=1)),
        "gt2": (true_flat > 2, probs_flat[:, 3]),
    }
    for name, (y_bin, pos_scores) in thresholds_zone.items():
        prefix = f"zone_bin_{name}"
        metrics_zone.update(_binary_block(prefix, y_bin.astype(np.int32), pos_scores))

    metrics_patient: dict[str, list[float] | list[int]] = {}
    metrics_region_pz: dict[str, list[float] | list[int]] = {}
    metrics_region_tz: dict[str, list[float] | list[int]] = {}

    hard_zone_regions = _coerce_hard_zone_regions(
        n_cases=n_cases,
        n_zones=n_zones,
        zone_regions=zone_regions,
        zone_region_weights=zone_region_weights,
    )

    for case_idx in range(n_cases):
        case_labels = true_labels[case_idx]
        case_probs = pred_probs[case_idx]
        case_regions = hard_zone_regions[case_idx]

        label_agg = int(case_labels.max())
        pos_maps = {
            "gt0": case_probs[:, 1:].sum(axis=1),
            "gt1": case_probs[:, 2:].sum(axis=1),
            "gt2": case_probs[:, 3],
        }
        agg_scores = {k: float(v.max()) for k, v in pos_maps.items()}
        for name, thr_val in (("gt0", 0), ("gt1", 1), ("gt2", 2)):
            metrics_patient.setdefault(f"patient_bin_{name}_y", []).append(int(label_agg > thr_val))
            metrics_patient.setdefault(f"patient_bin_{name}_score", []).append(agg_scores[name])

        if not _has_both_regions(case_regions):
            continue

        for region_key, metrics_store in (("pz", metrics_region_pz), ("tz", metrics_region_tz)):
            mask = _region_mask(case_regions, region_key)
            region_labels = case_labels[mask]
            region_probs = case_probs[mask]

            label_region = int(region_labels.max())
            pos_maps_region = {
                "gt0": region_probs[:, 1:].sum(axis=1),
                "gt1": region_probs[:, 2:].sum(axis=1),
                "gt2": region_probs[:, 3],
            }
            agg_region = {k: float(v.max()) for k, v in pos_maps_region.items()}

            for name, thr_val in (("gt0", 0), ("gt1", 1), ("gt2", 2)):
                metrics_store.setdefault(f"{region_key}_bin_{name}_y", []).append(int(label_region > thr_val))
                metrics_store.setdefault(f"{region_key}_bin_{name}_score", []).append(agg_region[name])


    def _finalize_from_store(store: dict[str, list[float] | list[int]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in list(store.keys()):
            if key.endswith("_y"):
                name = key[:-2]
                y_true = np.asarray(store[key])
                pos_scores = np.asarray(store[f"{name}_score"])
                out.update(_binary_block(name, y_true.astype(np.int32), pos_scores))
        return out

    patient_metrics_final = _finalize_from_store(metrics_patient)
    pz_metrics_final = _finalize_from_store(metrics_region_pz)
    tz_metrics_final = _finalize_from_store(metrics_region_tz)

    return {
        "zone": metrics_zone,
        "patient": patient_metrics_final,
        "pz": pz_metrics_final,
        "tz": tz_metrics_final,
    }


def classification_metrics(
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    zone_regions: np.ndarray | None = None,
    zone_region_weights: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Compute classification evaluation metrics for the fixed isup4 setup."""
    n_classes = pred_probs.shape[-1]

    if n_classes != 4:
        raise ValueError(f"isup4 expects 4 classes, got {n_classes}.")
    if true_labels.ndim != 2 or pred_probs.ndim != 3:
        raise ValueError(
            f"Expected true_labels (n_cases, n_zones) and pred_probs (n_cases, n_zones, n_classes), "
            f"got true_labels{true_labels.shape}, pred_probs{pred_probs.shape}."
        )
    if true_labels.shape[1] != 20 or pred_probs.shape[1] != 20:
        raise ValueError(
            f"Expected exactly 20 zones per case, got true_labels.shape[1]={true_labels.shape[1]}, "
            f"pred_probs.shape[1]={pred_probs.shape[1]}."
        )
    return _isup4_zone_metrics(
        true_labels=true_labels,
        pred_probs=pred_probs,
        zone_regions=zone_regions,
        zone_region_weights=zone_region_weights,
    )
