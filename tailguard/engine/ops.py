"""Training losses and evaluation operations required by TailGuard."""

import math
from functools import partial
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy import ndarray
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from skimage import measure
from torch.optim.lr_scheduler import _LRScheduler


def _scale_gradient(values, indices, factor=0.0):
    indices = indices.expand_as(values)
    values[indices] *= factor
    return values


def global_cosine_hm_percent(encoder_features, decoder_features, p=0.9, factor=0.0, sample_weight=None):
    """Dinomaly hard-mining cosine loss used by the reconstruction core."""
    cosine = torch.nn.CosineSimilarity()
    loss = 0
    if sample_weight is not None:
        sample_weight = sample_weight.reshape(-1).to(encoder_features[0].device)

    for encoder_feature, decoder_feature in zip(encoder_features, decoder_features):
        encoder_feature = encoder_feature.detach()
        with torch.no_grad():
            point_distance = 1 - cosine(encoder_feature, decoder_feature).unsqueeze(1)
            topk_count = max(1, int(point_distance.numel() * (1 - p)))
            threshold = torch.topk(point_distance.reshape(-1), k=topk_count)[0][-1]

        sample_loss = 1 - cosine(
            encoder_feature.reshape(encoder_feature.shape[0], -1),
            decoder_feature.reshape(decoder_feature.shape[0], -1),
        )
        if sample_weight is None:
            loss += torch.mean(sample_loss)
        else:
            loss += torch.sum(sample_weight * sample_loss) / sample_weight.sum().clamp_min(1e-12)

        decoder_feature.register_hook(
            partial(_scale_gradient, indices=point_distance < threshold, factor=factor)
        )

    return loss / len(encoder_features)


def _anomaly_maps(encoder_features, decoder_features, output_size):
    if not isinstance(output_size, tuple):
        output_size = (output_size, output_size)
    maps = []
    for encoder_feature, decoder_feature in zip(encoder_features, decoder_features):
        anomaly_map = 1 - F.cosine_similarity(encoder_feature, decoder_feature)
        anomaly_map = F.interpolate(
            anomaly_map.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
        maps.append(anomaly_map)
    return torch.cat(maps, dim=1).mean(dim=1, keepdim=True)


def f1_score_max(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    values = 2 * precision * recall / (precision + recall + 1e-7)
    return values[:-1].max()


def compute_image_level_scores(anomaly_map, max_ratio=0):
    if max_ratio == 0:
        return torch.max(anomaly_map.flatten(1), dim=1)[0]
    flat_map = anomaly_map.flatten(1)
    topk = max(1, int(flat_map.shape[1] * max_ratio))
    return torch.sort(flat_map, dim=1, descending=True)[0][:, :topk].mean(dim=1)


def infer_anomaly_map_batch(model, images, gaussian_kernel, resize_mask=None, gt=None, apply_smoothing=True):
    encoder_features, decoder_features = model(images)[:2]
    anomaly_map = _anomaly_maps(encoder_features, decoder_features, images.shape[-1])
    if resize_mask is not None:
        anomaly_map = F.interpolate(
            anomaly_map,
            size=resize_mask,
            mode="bilinear",
            align_corners=False,
        )
        if gt is not None:
            gt = F.interpolate(gt, size=resize_mask, mode="nearest")
    if apply_smoothing:
        anomaly_map = gaussian_kernel(anomaly_map)
    return anomaly_map, gt


@torch.no_grad()
def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None):
    del _class_
    model.eval()
    ground_truth_pixels = []
    predicted_pixels = []
    ground_truth_images = []
    predicted_images = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    for images, gt, labels, _ in dataloader:
        images = images.to(device)
        anomaly_map, gt = infer_anomaly_map_batch(
            model,
            images,
            gaussian_kernel,
            resize_mask=resize_mask,
            gt=gt,
        )
        gt = gt.bool()
        if gt.shape[1] > 1:
            gt = torch.max(gt, dim=1, keepdim=True)[0]
        ground_truth_pixels.append(gt)
        predicted_pixels.append(anomaly_map)
        ground_truth_images.append(labels)
        predicted_images.append(compute_image_level_scores(anomaly_map, max_ratio=max_ratio))

    gt_pixels = torch.cat(ground_truth_pixels, dim=0)[:, 0].cpu().numpy()
    pred_pixels = torch.cat(predicted_pixels, dim=0)[:, 0].cpu().numpy()
    gt_images = torch.cat(ground_truth_images).flatten().cpu().numpy()
    pred_images = torch.cat(predicted_images).flatten().cpu().numpy()

    aupro_pixels = compute_pro(gt_pixels, pred_pixels)
    flat_gt_pixels = gt_pixels.ravel()
    flat_pred_pixels = pred_pixels.ravel()
    return [
        roc_auc_score(gt_images, pred_images),
        average_precision_score(gt_images, pred_images),
        f1_score_max(gt_images, pred_images),
        roc_auc_score(flat_gt_pixels, flat_pred_pixels),
        average_precision_score(flat_gt_pixels, flat_pred_pixels),
        f1_score_max(flat_gt_pixels, flat_pred_pixels),
        aupro_pixels,
    ]


def compute_pro(masks: ndarray, anomaly_maps: ndarray, num_th: int = 200):
    """Compute AUPRO for false-positive rates below 0.3."""
    if not isinstance(anomaly_maps, ndarray) or not isinstance(masks, ndarray):
        raise TypeError("masks and anomaly_maps must be numpy arrays")
    if anomaly_maps.ndim != 3 or masks.ndim != 3 or anomaly_maps.shape != masks.shape:
        raise ValueError("masks and anomaly_maps must have identical [N,H,W] shapes")
    if not set(np.unique(masks)).issubset({0, 1}):
        raise ValueError("masks must be binary")

    minimum = float(anomaly_maps.min())
    maximum = float(anomaly_maps.max())
    if maximum <= minimum:
        return 0.0
    rows = []
    binary_maps = np.zeros_like(anomaly_maps, dtype=bool)
    for threshold in np.linspace(minimum, maximum, num=num_th, endpoint=False):
        binary_maps[:] = anomaly_maps > threshold
        overlaps = []
        for binary_map, mask in zip(binary_maps, masks):
            for region in measure.regionprops(measure.label(mask)):
                coordinates = region.coords
                overlaps.append(binary_map[coordinates[:, 0], coordinates[:, 1]].sum() / region.area)
        false_positive_pixels = np.logical_and(1 - masks, binary_maps).sum()
        negative_pixels = (1 - masks).sum()
        rows.append(
            {
                "pro": mean(overlaps) if overlaps else 0.0,
                "fpr": false_positive_pixels / max(1, negative_pixels),
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame[frame["fpr"] < 0.3]
    if len(frame) < 2 or frame["fpr"].max() <= 0:
        return 0.0
    frame["fpr"] = frame["fpr"] / frame["fpr"].max()
    return auc(frame["fpr"], frame["pro"])


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    coordinates = torch.arange(kernel_size)
    x_grid = coordinates.repeat(kernel_size).view(kernel_size, kernel_size)
    xy_grid = torch.stack([x_grid, x_grid.t()], dim=-1).float()
    center = (kernel_size - 1) / 2.0
    variance = sigma ** 2
    kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - center) ** 2.0, dim=-1) / (2.0 * variance)
    )
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    layer = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=kernel_size // 2,
    )
    layer.weight.data = kernel
    layer.weight.requires_grad = False
    return layer


class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iterations = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iterations / len(iterations))
        )
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        return [self.schedule[self.last_epoch] for _ in self.base_lrs]
