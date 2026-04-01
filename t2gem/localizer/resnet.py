from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer, get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option
from monai.networks.nets.resnet import get_inplanes, get_avgpool

def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError(f"Expected positive value for softplus inverse, got {value}.")
    tensor_value = torch.tensor(value, dtype=torch.float32)
    return float(torch.log(torch.expm1(tensor_value)).item())


def _as_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.to(dtype=torch.bool)


class ResNetBlock(nn.Module):
    """MONAI-compatible residual block without inplace residual addition."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: nn.Module | partial | None = None,
        act: str | tuple = ("relu", {"inplace": False}),
        norm: str | tuple = "batch",
    ) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.act = get_act_layer(name=act)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.act(out)

        return out


class ResNetBottleneck(nn.Module):
    """MONAI-compatible bottleneck block without inplace residual addition."""

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: nn.Module | partial | None = None,
        act: str | tuple = ("relu", {"inplace": False}),
        norm: str | tuple = "batch",
    ) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_layer = partial(get_norm_layer, name=norm, spatial_dims=spatial_dims)

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(channels=planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(channels=planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(channels=planes * self.expansion)
        self.act = get_act_layer(name=act)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.act(out)

        return out


class CRF(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_iter: int,
        sigma: float,
        smoothness_init: float,
        knn_k: int,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}.")
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}.")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}.")
        if smoothness_init <= 0:
            raise ValueError(f"smoothness_init must be > 0, got {smoothness_init}.")
        if knn_k < 1:
            raise ValueError(f"knn_k must be >= 1, got {knn_k}.")

        self.num_classes = num_classes
        self.n_iter = n_iter
        self.knn_k = knn_k

        sigma_tensor = torch.tensor(_inverse_softplus(float(sigma)), dtype=torch.float32)
        self.raw_sigma = nn.Parameter(sigma_tensor)
        self.raw_smoothness = nn.Parameter(
            torch.tensor(_inverse_softplus(float(smoothness_init)), dtype=torch.float32)
        )

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma).clamp_min(1e-6)

    @property
    def smoothness(self) -> torch.Tensor:
        return F.softplus(self.raw_smoothness)

    def _compatibility_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.arange(self.num_classes, device=device, dtype=dtype)
        diff = idx[:, None] - idx[None, :]
        compat = diff.square()
        max_val = compat.max().clamp_min(1.0)
        return compat / max_val

    def _build_adjacency(
        self,
        dist_sq: torch.Tensor,
        valid_pairs: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = dist_sq.shape[-1]
        k_eff = min(self.knn_k, max(n_nodes - 1, 1))
        masked_dist = dist_sq.masked_fill(~valid_pairs, float("inf")).masked_fill(identity, float("inf"))
        knn_dist, knn_idx = torch.topk(masked_dist, k=k_eff, dim=-1, largest=False)
        adjacency = torch.zeros_like(valid_pairs)
        adjacency.scatter_(2, knn_idx, torch.isfinite(knn_dist))
        adjacency = adjacency | adjacency.transpose(1, 2)
        adjacency = adjacency & valid_pairs & (~identity)
        return adjacency

    def _appearance_affinity(
        self,
        logits: torch.Tensor,
        appearance_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if appearance_features is None:
            raise ValueError("appearance_features must be provided for CRF appearance affinity.")
        feats = F.normalize(appearance_features.to(dtype=logits.dtype), p=2, dim=-1, eps=1e-6)
        cosine = torch.bmm(feats, feats.transpose(1, 2))
        return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        rois: torch.Tensor,
        appearance_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f"logits must have shape (B, N, C), got {tuple(logits.shape)}.")
        if rois.shape[:2] != logits.shape[:2] or rois.shape[-1] != 6:
            raise ValueError(f"Expected rois shape (B, N, 6) matching logits, got {tuple(rois.shape)}.")
        _, n_nodes, n_classes = logits.shape
        if n_classes != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {n_classes}.")
        if n_nodes <= 1:
            return logits

        roi_centers = (rois[:, :, :3] + rois[:, :, 3:]) / 2.0
        dist_sq = torch.cdist(roi_centers.to(dtype=logits.dtype), roi_centers.to(dtype=logits.dtype), p=2).pow(2)
        identity = torch.eye(n_nodes, device=logits.device, dtype=torch.bool).unsqueeze(0)
        valid_mask = _as_bool_mask(rois.abs().sum(dim=-1) > 0)
        valid_pairs = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        adjacency = self._build_adjacency(dist_sq=dist_sq, valid_pairs=valid_pairs, identity=identity)

        sigma = self.sigma.to(dtype=logits.dtype)
        smoothness = self.smoothness.to(dtype=logits.dtype)
        pairwise_kernel = torch.exp(-0.5 * dist_sq / sigma.square())
        appearance_affinity = self._appearance_affinity(logits=logits, appearance_features=appearance_features)
        pairwise_kernel = pairwise_kernel * appearance_affinity
        pairwise_kernel = pairwise_kernel.masked_fill(identity, 0.0)
        pairwise_kernel = pairwise_kernel * adjacency.to(dtype=logits.dtype)
        pairwise_kernel = pairwise_kernel / pairwise_kernel.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        q_probs = F.softmax(logits, dim=-1)
        compat = self._compatibility_matrix(device=logits.device, dtype=logits.dtype)
        refined_logits = logits
        for _ in range(self.n_iter):
            q_neighbors = torch.bmm(pairwise_kernel, q_probs)
            pairwise_term = torch.einsum("bnc,cd->bnd", q_neighbors, compat)
            refined_logits = logits - smoothness * pairwise_term
            q_probs = F.softmax(refined_logits, dim=-1)
        return refined_logits


class ResNet(nn.Module):
    """Minimal 3D ResNet wrapper."""

    def __init__(
        self,
        block: type[ResNetBlock | ResNetBottleneck] | str,
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: tuple[int] | int = 7,
        conv1_t_stride: tuple[int] | int = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        bias_downsample: bool = True, 
        act: str | tuple = ("relu", {"inplace": False}),
        norm: str | tuple = "batch",
    ) -> None:
        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avgp_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample
        self.act_cfg = act
        self.norm_cfg = norm

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=tuple(k // 2 for k in conv1_kernel_size),
            bias=False,
        )

        norm_layer = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=self.in_planes)
        self.bn1 = norm_layer
        self.act = get_act_layer(name=act)
        self.block_expansion = block.expansion
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], spatial_dims, shortcut_type, act=self.act_cfg, norm=self.norm_cfg
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2, act=self.act_cfg, norm=self.norm_cfg
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2, act=self.act_cfg, norm=self.norm_cfg
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2, act=self.act_cfg, norm=self.norm_cfg
        )
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, type(norm_layer)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: type[ResNetBlock | ResNetBottleneck],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
        act: str | tuple = ("relu", {"inplace": False}),
        norm: str | tuple = "batch",
    ) -> nn.Sequential:
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        downsample: nn.Module | partial | None = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
                    get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes,
                planes=planes,
                spatial_dims=spatial_dims,
                stride=stride,
                downsample=downsample,
                act=act,
                norm=norm,
            )
        ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims, act=act, norm=norm))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


def generate_model(model_depth, n_input_channels, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(ResNetBlock, [1, 1, 1, 1], get_inplanes(), spatial_dims=3, n_input_channels=n_input_channels, shortcut_type='B', **kwargs)
    elif model_depth == 18:
        model = ResNet(ResNetBlock, [2, 2, 2, 2], get_inplanes(), spatial_dims=3, n_input_channels=n_input_channels, shortcut_type='B', **kwargs)
    elif model_depth == 34:
        model = ResNet(ResNetBlock, [3, 4, 6, 3], get_inplanes(), spatial_dims=3, n_input_channels=n_input_channels, shortcut_type='B', **kwargs)
    elif model_depth == 50:
        model = ResNet(ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), spatial_dims=3, n_input_channels=n_input_channels, shortcut_type='B', **kwargs)
    elif model_depth == 101:
        model = ResNet(ResNetBottleneck, [3, 4, 23, 3], get_inplanes(), spatial_dims=3, n_input_channels=n_input_channels, shortcut_type='B', **kwargs)
    elif model_depth == 152:
        model = ResNet(ResNetBottleneck, [3, 8, 36, 3], get_inplanes(), spatial_dims=3, n_input_channels=n_input_channels, shortcut_type='B', **kwargs)
    elif model_depth == 200:
        model = ResNet(ResNetBottleneck, [3, 24, 36, 3], get_inplanes(), spatial_dims=3, n_input_channels=n_input_channels, shortcut_type='B', **kwargs)

    return model

class FPN3D(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN3D, self).__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

        for m in list(self.lateral_convs) + list(self.output_convs):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, features):
        last_inner = self.lateral_convs[-1](features[-1])
        results = [self.output_convs[-1](last_inner)]

        for i in range(len(features) - 2, -1, -1):
            lateral_feature = self.lateral_convs[i](features[i])
            top_down_feature = F.interpolate(last_inner, size=lateral_feature.shape[2:], mode="trilinear", align_corners=False)
            last_inner = lateral_feature + top_down_feature
            results.insert(0, self.output_convs[i](last_inner))

        return results

class BackboneWithFPN(nn.Module):
    def __init__(self, model_depth, n_input_channels, out_channels=256):
        super(BackboneWithFPN, self).__init__()
        
        # ResNet50
        self.resnet = generate_model(model_depth, n_input_channels, conv1_t_stride=2)
        expansion = self.resnet.block_expansion
        
        in_channels_list = [
            64 * expansion,    # C2
            128 * expansion,   # C3
            256 * expansion,   # C4
            512 * expansion    # C5
        ]
        self.spatial_strides = [4, 8, 16, 32]
        self.spatial_scales = [1.0 / s for s in self.spatial_strides]
        
        # FPN
        self.fpn = FPN3D(in_channels_list, out_channels)
        
    def forward(self, x):
        features = self.resnet(x)
        return self.fpn(features)

class ROIAlign3D(nn.Module):
    """3D ROI Align based on grid_sample."""

    def __init__(self, output_size, spatial_scale=1.0, sampling_ratio=2, aligned=True):
        super(ROIAlign3D, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, features: torch.Tensor, rois: torch.Tensor, spatial_scale: float = None) -> torch.Tensor:
        batch_size, num_rois = rois.shape[:2]
        _, C, D, H, W = features.shape
        out_d, out_h, out_w = self.output_size
        scale = self.spatial_scale if spatial_scale is None else spatial_scale

        output = features.new_zeros((batch_size, num_rois, C, out_d, out_h, out_w))

        sd = sh = sw = self.sampling_ratio if self.sampling_ratio > 0 else 1
        
        d_step = (torch.arange(out_d * sd, device=features.device, dtype=features.dtype) + 0.5) / sd
        h_step = (torch.arange(out_h * sh, device=features.device, dtype=features.dtype) + 0.5) / sh
        w_step = (torch.arange(out_w * sw, device=features.device, dtype=features.dtype) + 0.5) / sw

        for b in range(batch_size):
            cur_rois = rois[b] * scale
            if self.aligned: cur_rois -= 0.5
            
            z1, y1, x1 = cur_rois[:, 0:1], cur_rois[:, 1:2], cur_rois[:, 2:3]
            z2, y2, x2 = cur_rois[:, 3:4], cur_rois[:, 4:5], cur_rois[:, 5:6]
            
            roi_d = (z2 - z1).clamp_min(1e-6) # 1e-6 for safety
            roi_h = (y2 - y1).clamp_min(1e-6)
            roi_w = (x2 - x1).clamp_min(1e-6)
            
            bin_d, bin_h, bin_w = roi_d / out_d, roi_h / out_h, roi_w / out_w
            
            z_grid = z1 + d_step.unsqueeze(0) * bin_d
            y_grid = y1 + h_step.unsqueeze(0) * bin_h
            x_grid = x1 + w_step.unsqueeze(0) * bin_w
            
            z_norm = (z_grid + 0.5) / D * 2 - 1
            y_norm = (y_grid + 0.5) / H * 2 - 1
            x_norm = (x_grid + 0.5) / W * 2 - 1
            
            N = num_rois
            D_pts, H_pts, W_pts = out_d*sd, out_h*sh, out_w*sw
            
            g_z = z_norm.view(N, D_pts, 1, 1).expand(N, D_pts, H_pts, W_pts)
            g_y = y_norm.view(N, 1, H_pts, 1).expand(N, D_pts, H_pts, W_pts)
            g_x = x_norm.view(N, 1, 1, W_pts).expand(N, D_pts, H_pts, W_pts)
            
            grid = torch.stack([g_x, g_y, g_z], dim=-1) # [N, Dp, Hp, Wp, 3]
            grid = grid.reshape(1, N * D_pts, H_pts, W_pts, 3) 
            
            cur_feat = features[b:b+1] # [1, C, D, H, W]
            sampled = F.grid_sample(cur_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled = sampled.view(1, C, N, out_d, sd, out_h, sh, out_w, sw)
            output[b] = sampled.mean(dim=(4, 6, 8)).squeeze(0).permute(1, 0, 2, 3, 4)

        return output


class RoIHead(nn.Module):
    def __init__(self, in_channels=256, roi_size=(7,7,7), num_classes=10, dropout=0.5):
        super().__init__()
        self.embedding_dim = 256
        self.pool = nn.Sequential(
            nn.Conv3d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, roi_features, return_embed: bool = False):
        B, N = roi_features.shape[:2]
        if N == 0:
            out_logits = roi_features.new_zeros((B, 0, self.fc[-1].out_features))
            out_embed = roi_features.new_zeros((B, 0, self.embedding_dim))
            return (out_logits, out_embed) if return_embed else out_logits
        x = roi_features.reshape(B * N, *roi_features.shape[2:])
        x = self.pool(x)
        x = self.fc[0](x)
        x = self.fc[1](x)
        x = self.fc[2](x)
        embed = self.fc[3](x)
        logits = self.fc[4](embed)

        logits = logits.view(B, N, -1)
        embed = embed.view(B, N, -1)
        if return_embed:
            return logits, embed
        return logits

class RoINet(nn.Module):
    def __init__(self, model_depth=50, in_channels=2, roi_size=(7,7,7), roi_classes=2, sampling_ratio=2,
                 canonical_scale=64, canonical_level=3, use_crf=False,
                 crf_n_iter=2, crf_sigma_init=10.0, crf_smoothness_init=0.2, crf_knn_k=3):
        super().__init__()
        self.backbone = BackboneWithFPN(model_depth, in_channels)
        self.min_level = 2
        self.max_level = self.min_level + len(self.backbone.spatial_scales) - 1
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        self.poolers = nn.ModuleList([
            ROIAlign3D(
                output_size=roi_size,
                spatial_scale=scale,
                sampling_ratio=sampling_ratio
            ) for scale in self.backbone.spatial_scales
        ])
        self.roi_head = RoIHead(in_channels=256, num_classes=roi_classes)
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(
                num_classes=roi_classes,
                n_iter=crf_n_iter,
                sigma=crf_sigma_init,
                smoothness_init=crf_smoothness_init,
                knn_k=crf_knn_k,
            )

    def _pool_rois(self, feats, rois):
        B, N = rois.shape[:2]
        aligned_feats = feats[0].new_zeros((B, N, feats[0].shape[1], *self.poolers[0].output_size))

        for b in range(B):
            rois_b = rois[b]
            roi_sizes = (rois_b[:, 3:6] - rois_b[:, 0:3]).clamp_min(1e-6)
            roi_scales = roi_sizes.prod(dim=1).pow(1/3)
            target_lvls = (self.canonical_level + torch.log2(roi_scales / self.canonical_scale + 1e-6)).floor()
            target_lvls = target_lvls.clamp(self.min_level, self.max_level).to(torch.int64)

            for lvl_idx, pooler in enumerate(self.poolers):
                target_level = self.min_level + lvl_idx
                idxs = torch.nonzero(target_lvls == target_level, as_tuple=False).squeeze(1)
                if idxs.numel() == 0:
                    continue

                rois_lvl = rois_b[idxs].unsqueeze(0)  # [1, num_rois_level, 6]
                feat_lvl = feats[lvl_idx][b:b+1]
                pooled = pooler(feat_lvl, rois_lvl)  # [1, num_rois_level, C, d, h, w]
                aligned_feats[b, idxs] = pooled[0]
        return aligned_feats

    def forward(self, x, rois, return_features: bool = False):
        feats = self.backbone(x)
        B, N = rois.shape[:2]
        aligned_feats = self._pool_rois(feats, rois)

        need_embed = return_features or self.use_crf
        if need_embed:
            roi_pred, roi_embed = self.roi_head(aligned_feats, return_embed=True)
        else:
            roi_pred = self.roi_head(aligned_feats)
            roi_embed = None
        if self.use_crf and N > 1:
            roi_pred = self.crf(roi_pred, rois, appearance_features=roi_embed)
        if return_features:
            return roi_pred, roi_embed
        return roi_pred
