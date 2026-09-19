"""Tail candidate discovery used by the canonical TailGuard pipeline."""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed

# The paper environment uses an older SciPy build.  Restore the NumPy aliases
# that SciPy imports inside joblib worker processes after a NumPy upgrade.
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

from scipy.stats import mode


PLATEAU_GAP_GUARD_V1 = 'plateau_gap_v1'


def compute_self_sim(features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Return image-level cosine similarity for a two-dimensional feature set."""
    if features.ndim != 2:
        raise ValueError('tail sampler expects image-level features with shape [N, D]')
    if normalize:
        features = F.normalize(features, dim=-1)
    return torch.matmul(features, features.t())


def predict_num_samples_per_class(class_sizes: torch.Tensor,
                                  round_class_sizes: bool = True) -> torch.Tensor:
    """Convert per-image class-size estimates into an inferred class histogram."""
    if round_class_sizes:
        class_sizes = torch.round(class_sizes).to(torch.long)
    class_sizes_sorted = torch.sort(class_sizes, descending=False)[0]
    class_sizes_sorted = torch.maximum(
        class_sizes_sorted,
        torch.ones_like(class_sizes_sorted),
    ).reshape(-1)

    num_samples_per_class = []
    while len(class_sizes_sorted) > 0:
        if len(class_sizes_sorted) < class_sizes_sorted[0]:
            if len(num_samples_per_class) == 0:
                num_samples_per_class.append(len(class_sizes_sorted))
            else:
                num_samples_per_class[-1] += len(class_sizes_sorted)
            break

        num_samples = torch.round(
            class_sizes_sorted[:class_sizes_sorted[0]].float().mean()
        ).long().item()
        num_samples = min(num_samples, len(class_sizes_sorted))
        num_samples_per_class.append(num_samples)
        class_sizes_sorted = class_sizes_sorted[num_samples:]

    return torch.LongTensor(num_samples_per_class).sort(descending=True)[0]


def compute_orthogonal_distances(points: torch.Tensor,
                                 line_points: torch.Tensor) -> torch.Tensor:
    points_np = points.numpy()
    (x1, y1), (x2, y2) = line_points[0].numpy(), line_points[1].numpy()
    dx, dy = x2 - x1, y2 - y1
    if dx == 0:
        return torch.FloatTensor([abs(x - x1) for x, _ in points_np])
    if dy == 0:
        return torch.FloatTensor([abs(y - y1) for _, y in points_np])

    slope = dy / dx
    intercept = y1 - slope * x1
    perpendicular_slope = -1 / slope
    distances = []
    for x, y in points_np:
        perpendicular_intercept = y - perpendicular_slope * x
        x_intersect = (
            (perpendicular_intercept - intercept)
            / (slope - perpendicular_slope)
        )
        y_intersect = slope * x_intersect + intercept
        distances.append(
            np.sqrt((x_intersect - x) ** 2 + (y_intersect - y) ** 2)
        )
    return torch.FloatTensor(distances)


def elbow(scores: torch.Tensor, sort: bool = True, quantize: bool = False):
    if quantize:
        scale = float(len(scores))
        scores = (scores * scale).long()
    else:
        scale = 1.0
    if sort:
        scores = scores.sort(descending=True)[0]
    points = torch.cat(
        [torch.arange(len(scores))[:, None], scores[:, None]],
        dim=1,
    )
    distances = compute_orthogonal_distances(points, points[[0, -1], :])
    return scores[distances.argmax()].item() / scale


def _predict_max_k_within_percentile(num_samples_per_class: torch.Tensor,
                                     percentile: float = 0.15):
    ordered = num_samples_per_class.sort(descending=True)[0]
    sample_limit = int(ordered.sum() * percentile)
    cumulative_tail = ordered.flip(dims=[0]).cumsum(dim=0)
    num_tail_classes = 0
    for count in cumulative_tail:
        if count > sample_limit:
            break
        num_tail_classes += 1
    index = min(len(ordered) - num_tail_classes, len(ordered) - 1)
    return ordered[index].item()


def predict_max_k(num_samples_per_class: torch.Tensor,
                  percentile: float = 0.15):
    max_k = elbow(num_samples_per_class)
    if percentile < 1:
        max_k = min(
            max_k,
            _predict_max_k_within_percentile(
                num_samples_per_class,
                percentile=percentile,
            ),
        )
    return max_k


def _apply_plateau_gap_guard(class_sizes: torch.Tensor,
                             original_mask: torch.Tensor,
                             original_cutoff):
    class_sizes = class_sizes.reshape(-1)
    original_mask = original_mask.reshape(-1).to(
        device=class_sizes.device,
        dtype=torch.long,
    )
    num_samples = int(class_sizes.numel())
    num_selected_raw = int(original_mask.sum().item())
    diagnostics = {
        'guard_name': PLATEAU_GAP_GUARD_V1,
        'enabled': True,
        'triggered': False,
        'status': 'unchanged',
        'num_samples': num_samples,
        'num_selected_raw': num_selected_raw,
        'num_selected_final': num_selected_raw,
        'num_removed_by_guard': 0,
        'selected_ratio_raw': num_selected_raw / num_samples if num_samples else 0.0,
        'selected_ratio_final': num_selected_raw / num_samples if num_samples else 0.0,
        'original_cutoff': None if original_cutoff is None else float(original_cutoff),
        'effective_cutoff': None if original_cutoff is None else float(original_cutoff),
        'num_inferred_classes': 0,
        'boundary_multiplicity': 0,
        'gap_lower': None,
        'gap_upper': None,
        'dominant_log_gap': None,
    }
    if num_selected_raw == 0 or original_cutoff is None:
        diagnostics['status'] = 'unchanged_no_selection'
        return original_mask, diagnostics

    inferred_sizes = predict_num_samples_per_class(
        class_sizes.detach().cpu()
    ).sort()[0]
    diagnostics['num_inferred_classes'] = int(inferred_sizes.numel())
    if inferred_sizes.numel() < 2:
        diagnostics['status'] = 'unchanged_too_few_inferred_classes'
        return original_mask, diagnostics

    log_sizes = torch.log(inferred_sizes.float().clamp_min(1.0))
    log_gaps = log_sizes[1:] - log_sizes[:-1]
    gap_index = int(torch.argmax(log_gaps).item())
    gap_lower = int(inferred_sizes[gap_index].item())
    gap_upper = int(inferred_sizes[gap_index + 1].item())
    cutoff = int(round(float(original_cutoff)))
    boundary_multiplicity = int((inferred_sizes == cutoff).sum().item())
    diagnostics.update({
        'boundary_multiplicity': boundary_multiplicity,
        'gap_lower': float(gap_lower),
        'gap_upper': float(gap_upper),
        'dominant_log_gap': float(log_gaps[gap_index].item()),
    })
    if boundary_multiplicity < 2 or cutoff <= gap_lower:
        diagnostics['status'] = 'unchanged_supported_boundary'
        return original_mask, diagnostics

    guarded_mask = (class_sizes <= gap_lower).to(torch.long)
    num_selected_final = int(guarded_mask.sum().item())
    if num_selected_final > num_selected_raw:
        diagnostics['status'] = 'unchanged_non_monotonic_guard'
        return original_mask, diagnostics
    diagnostics.update({
        'triggered': True,
        'status': 'triggered',
        'num_selected_final': num_selected_final,
        'num_removed_by_guard': num_selected_raw - num_selected_final,
        'selected_ratio_final': num_selected_final / num_samples if num_samples else 0.0,
        'effective_cutoff': float(gap_lower),
    })
    return guarded_mask, diagnostics


def predict_few_shot_class_samples(class_sizes: torch.Tensor,
                                   percentile: float = 0.15):
    inferred_sizes = predict_num_samples_per_class(class_sizes)
    max_k = predict_max_k(inferred_sizes, percentile=percentile)
    raw_mask = (class_sizes <= max_k).to(torch.long)
    return _apply_plateau_gap_guard(class_sizes, raw_mask, max_k)


def _sort(scores: np.ndarray, descending: bool = False) -> np.ndarray:
    dtype = scores.dtype
    scale = len(scores)
    scores = (scale * scores).astype(np.int_)
    scores = np.sort(scores).astype(dtype) / scale
    return scores[::-1] if descending else scores


def _compute_trim_min_threshold(scores: np.ndarray):
    scores = _sort(scores, descending=False)
    minimum = scores.min()
    half_angle_threshold = np.cos(np.arccos(minimum) / 2)
    upper_scores = scores[scores > half_angle_threshold]
    return upper_scores[-int(len(upper_scores) * 0.85)]


def _compute_ths(self_sim: torch.Tensor,
                 threshold_type: str = 'trim_min') -> torch.Tensor:
    """Compute the fixed per-image trim-min similarity thresholds."""
    if threshold_type != 'trim_min':
        raise ValueError('TailGuard uses only the trim_min threshold')
    similarity = self_sim.numpy()
    thresholds = Parallel(n_jobs=-1)(
        delayed(_compute_trim_min_threshold)(similarity[index])
        for index in range(len(similarity))
    )
    return torch.FloatTensor(thresholds)[:, None]


def _predict_adaptive_class_sizes(self_sim: torch.Tensor) -> torch.Tensor:
    thresholds = _compute_ths(self_sim)
    mask = self_sim >= thresholds
    count_map = mask * mask.sum(dim=1, keepdim=True)
    count_map = count_map.numpy()
    count_map = np.where(count_map == 0, np.nan, count_map)
    class_sizes = mode(count_map, axis=0, nan_policy='omit').mode
    return torch.from_numpy(class_sizes).to(torch.float)


def _sample_tail_candidates(features: torch.Tensor, percentile: float):
    self_sim = compute_self_sim(features)
    class_sizes = _predict_adaptive_class_sizes(self_sim).squeeze()
    selected_mask, diagnostics = predict_few_shot_class_samples(
        class_sizes,
        percentile=percentile,
    )
    selected_indices = torch.where(selected_mask == 1)[0]
    return features[selected_indices], selected_indices, class_sizes, diagnostics


class TailSampler:
    """The fixed adaptive TailSampler used to create initial tail candidates."""

    def __init__(self, percentile: float = 0.15):
        self.percentile = float(percentile)
        self.last_partition_diagnostics = None

    def run(self,
            features: torch.Tensor,
            feature_map_shape: Optional[torch.Tensor] = None,
            return_class_sizes: bool = False):
        if feature_map_shape is not None:
            raise ValueError('TailSampler expects image-level features')
        tail_samples, tail_indices, class_sizes, diagnostics = (
            _sample_tail_candidates(features, self.percentile)
        )
        self.last_partition_diagnostics = diagnostics
        if return_class_sizes:
            return tail_samples, tail_indices, class_sizes
        return tail_samples, tail_indices


def build_tail_sampler(percentile: float = 0.15):
    """Build the canonical TailGuard TailSampler."""
    return TailSampler(percentile=percentile), 'TailSampler(trim_min,mode)'
