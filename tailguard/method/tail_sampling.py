import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import mode
from sklearn.neighbors import KernelDensity


EPS = 1e-12
_WITHIN_FACTOR = 2
PLATEAU_GAP_GUARD_V1 = 'plateau_gap_v1'


def compute_self_sim(features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    if normalize:
        features = F.normalize(features, dim=-1)

    if features.ndim == 2:
        features_t = features.t()
    else:
        raise ValueError('tail sampler expects image-level features with shape [N, D]')

    return torch.matmul(features, features_t)


def predict_class_sizes(self_sim: torch.Tensor, th, vote_type: str = 'mean') -> torch.Tensor:
    batch_size = self_sim.shape[0]
    mask = self_sim >= th
    count_map = mask * mask.sum(dim=1, keepdim=True)

    if vote_type == 'mean':
        class_sizes = count_map.sum(dim=0) / torch.count_nonzero(count_map, dim=0)
    elif vote_type == 'mode':
        count_map = count_map.numpy()
        count_map_naned = np.where(count_map == 0, np.nan, count_map)
        class_sizes = mode(count_map_naned, axis=0, nan_policy='omit').mode
    elif vote_type == 'nearest':
        columnwise_means = count_map.sum(dim=0, keepdim=True) / torch.count_nonzero(count_map, dim=0)[None, :]
        idxes = torch.argmin((count_map - columnwise_means).abs(), dim=0)
        class_sizes = count_map[torch.arange(batch_size), idxes]
    else:
        raise ValueError(f'unsupported vote_type: {vote_type}')

    if isinstance(class_sizes, np.ndarray):
        class_sizes = torch.from_numpy(class_sizes)

    return class_sizes.to(torch.float)


def predict_num_samples_per_class(class_sizes: torch.Tensor, round_class_sizes: bool = True) -> torch.Tensor:
    if round_class_sizes:
        class_sizes = torch.round(class_sizes).to(torch.long)
    class_sizes_sorted = torch.sort(class_sizes, descending=False)[0]
    class_sizes_sorted = torch.maximum(class_sizes_sorted, torch.ones_like(class_sizes_sorted))
    class_sizes_sorted = class_sizes_sorted.reshape(-1)

    num_samples_per_class = []
    while len(class_sizes_sorted) > 0:
        if len(class_sizes_sorted) < class_sizes_sorted[0]:
            if len(num_samples_per_class) == 0:
                num_samples_per_class.append(len(class_sizes_sorted))
            else:
                num_samples_per_class[-1] += len(class_sizes_sorted)
            break

        num_samples_in_current_class = class_sizes_sorted[0]
        num_samples_in_current_class = (
            torch.round(class_sizes_sorted[:num_samples_in_current_class].float().mean()).long().item()
        )
        num_samples_in_current_class = min(num_samples_in_current_class, len(class_sizes_sorted))
        num_samples_per_class.append(num_samples_in_current_class)
        class_sizes_sorted = class_sizes_sorted[num_samples_in_current_class:]

    return torch.LongTensor(num_samples_per_class).sort(descending=True)[0]


def _apply_plateau_gap_guard(class_sizes: torch.Tensor,
                             original_mask: torch.Tensor,
                             original_cutoff,
                             guard_name: Optional[str]):
    class_sizes = class_sizes.reshape(-1)
    original_mask = original_mask.reshape(-1).to(
        device=class_sizes.device,
        dtype=torch.long,
    )
    num_samples = int(class_sizes.numel())
    num_selected_raw = int(original_mask.sum().item())
    diagnostics = {
        'guard_name': guard_name,
        'enabled': guard_name is not None,
        'triggered': False,
        'status': 'disabled' if guard_name is None else 'unchanged',
        'num_samples': num_samples,
        'num_selected_raw': num_selected_raw,
        'num_selected_final': num_selected_raw,
        'num_removed_by_guard': 0,
        'selected_ratio_raw': num_selected_raw / num_samples if num_samples > 0 else 0.0,
        'selected_ratio_final': num_selected_raw / num_samples if num_samples > 0 else 0.0,
        'original_cutoff': None if original_cutoff is None else float(original_cutoff),
        'effective_cutoff': None if original_cutoff is None else float(original_cutoff),
        'num_inferred_classes': 0,
        'boundary_multiplicity': 0,
        'gap_lower': None,
        'gap_upper': None,
        'dominant_log_gap': None,
    }
    if guard_name is None:
        return original_mask, diagnostics
    if guard_name != PLATEAU_GAP_GUARD_V1:
        raise ValueError(f'unsupported tail partition guard: {guard_name}')
    if num_selected_raw == 0 or original_cutoff is None:
        diagnostics['status'] = 'unchanged_no_selection'
        return original_mask, diagnostics

    inferred_class_sizes = predict_num_samples_per_class(
        class_sizes.detach().cpu()
    ).sort()[0]
    diagnostics['num_inferred_classes'] = int(inferred_class_sizes.numel())
    if inferred_class_sizes.numel() < 2:
        diagnostics['status'] = 'unchanged_too_few_inferred_classes'
        return original_mask, diagnostics

    log_class_sizes = torch.log(inferred_class_sizes.float().clamp_min(1.0))
    log_gaps = log_class_sizes[1:] - log_class_sizes[:-1]
    gap_index = int(torch.argmax(log_gaps).item())
    gap_lower = int(inferred_class_sizes[gap_index].item())
    gap_upper = int(inferred_class_sizes[gap_index + 1].item())
    cutoff_value = int(round(float(original_cutoff)))
    boundary_multiplicity = int((inferred_class_sizes == cutoff_value).sum().item())
    diagnostics.update({
        'boundary_multiplicity': boundary_multiplicity,
        'gap_lower': float(gap_lower),
        'gap_upper': float(gap_upper),
        'dominant_log_gap': float(log_gaps[gap_index].item()),
    })

    if boundary_multiplicity < 2 or cutoff_value <= gap_lower:
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
        'selected_ratio_final': num_selected_final / num_samples if num_samples > 0 else 0.0,
        'effective_cutoff': float(gap_lower),
    })
    return guarded_mask, diagnostics


def predict_few_shot_class_samples(class_sizes: torch.Tensor,
                                   percentile: float = 0.15,
                                   partition_guard: Optional[str] = None,
                                   return_partition_diagnostics: bool = False):
    num_samples_per_class = predict_num_samples_per_class(class_sizes)
    max_k = predict_max_k(num_samples_per_class, percentile=percentile)
    original_mask = (class_sizes <= max_k).to(torch.long)
    selected_mask, diagnostics = _apply_plateau_gap_guard(
        class_sizes,
        original_mask,
        max_k,
        partition_guard,
    )
    if return_partition_diagnostics:
        return selected_mask, diagnostics
    return selected_mask


def predict_max_k(num_samples_per_class: torch.Tensor, percentile: float = 0.15):
    max_k = _predict_max_k_elbow(num_samples_per_class)
    if percentile < 1:
        max_k_percentile = _predict_max_k_max_within_percnetile(num_samples_per_class, p=percentile)
        max_k = min(max_k, max_k_percentile)
    return max_k


def _predict_max_k_max_within_percnetile(num_samples_per_class: torch.Tensor, p: float = 0.15):
    num_samples_per_class = num_samples_per_class.sort(descending=True)[0]
    n = int(num_samples_per_class.sum() * p)
    cum_den = num_samples_per_class.flip(dims=[0]).cumsum(dim=0)

    k = 0
    for nspc in cum_den:
        if nspc > n:
            break
        k += 1

    max_k_idx = len(cum_den) - k
    max_k_idx = min(max_k_idx, len(num_samples_per_class) - 1)
    max_k = num_samples_per_class[max_k_idx]
    return max_k.item()


def _predict_max_k_elbow(num_samples_per_class: torch.Tensor):
    return elbow(num_samples_per_class)


def predict_few_shot_class_samples_by_scores(scores: torch.Tensor) -> torch.Tensor:
    elbow_score = elbow(scores)
    return (scores < elbow_score).long()


def elbow(scores: torch.Tensor, sort: bool = True, quantize: bool = False):
    if quantize:
        n = float(len(scores))
        scores = (scores * n).long()
    else:
        n = 1.0

    if sort:
        scores = scores.sort(descending=True)[0]

    score_points = torch.cat(
        [
            torch.arange(len(scores))[:, None],
            scores[:, None],
        ],
        dim=1,
    )
    orthogonal_distances = compute_orthogonal_distances(score_points, score_points[[0, -1], :])
    elbow_idx = orthogonal_distances.argmax()
    return scores[elbow_idx].item() / n


def compute_self_sim_min(self_sim: torch.Tensor, mode: str = 'min') -> torch.Tensor:
    mins = torch.empty(len(self_sim))
    for index in range(len(self_sim)):
        mins[index] = compute_min(self_sim[index])

    if mode == 'avg':
        minimum = mins.mean()
    elif mode == 'min':
        minimum = mins.min()
    else:
        raise ValueError(f'unsupported min mode: {mode}')

    return minimum


def compute_min(scores: torch.Tensor) -> torch.Tensor:
    return _compute_min(scores)


def _compute_min(scores: torch.Tensor) -> torch.Tensor:
    return scores.min()


def compute_sym_th(self_sim: torch.Tensor, mode: str = 'min', within: bool = False) -> float:
    factor = _WITHIN_FACTOR if within else 1
    minimum = compute_self_sim_min(self_sim, mode=mode)
    th = torch.cos(torch.acos(minimum) / (2 * factor)).item()
    return th


def _compute_th(self_sim: torch.Tensor, th_type: str = 'symmin', within: bool = False) -> float:
    factor = _WITHIN_FACTOR if within else 1

    if th_type == 'symmin':
        th = compute_sym_th(self_sim, mode='min', within=within)
    elif th_type == 'symavg':
        th = compute_sym_th(self_sim, mode='avg', within=within)
    elif th_type == 'indep':
        th = np.cos((np.pi / 2) / (2 * factor))
    else:
        raise ValueError(f'unsupported threshold_type: {th_type}')

    return th


def compute_th(
    self_sim: torch.Tensor,
    num_bootstrapping: int = 1,
    subsampling_ratio: float = 1.0,
    th_type: str = 'symmin',
    within: bool = False,
) -> float:
    if num_bootstrapping == 1:
        return _compute_th(self_sim, th_type=th_type, within=within)

    n = len(self_sim)
    batch = round(subsampling_ratio * n)
    ths = []
    for _ in range(num_bootstrapping):
        idxes = torch.randint(0, n, (batch,))
        sampled_self_sim = self_sim[idxes, :]
        sampled_self_sim = sampled_self_sim[:, idxes]
        ths.append(_compute_th(sampled_self_sim, th_type=th_type, within=within))
    return float(np.mean(ths))


def sample_few_shot(
    X: torch.Tensor,
    fea_map_shape=None,
    th_type: str = 'symmin',
    vote_type: str = 'mean',
    num_bootstrapping: int = 1,
    subsampling_ratio: float = 1.0,
    percentile: float = 0.15,
    return_class_sizes: bool = False,
):
    if fea_map_shape is not None:
        raise ValueError('tail sampler image-level port does not support feature_map_shape')
    return _sample_few_shot(
        X,
        th_type=th_type,
        vote_type=vote_type,
        num_bootstrapping=num_bootstrapping,
        subsampling_ratio=subsampling_ratio,
        percentile=percentile,
        return_class_sizes=return_class_sizes,
    )


def _sample_few_shot(
    X: torch.Tensor,
    th_type: str = 'symmin',
    vote_type: str = 'mean',
    num_bootstrapping: int = 1,
    subsampling_ratio: float = 1.0,
    percentile: float = 0.15,
    return_class_sizes: bool = False,
):
    self_sim = compute_self_sim(X, normalize=True)
    th = compute_th(
        self_sim,
        num_bootstrapping=num_bootstrapping,
        subsampling_ratio=subsampling_ratio,
        th_type=th_type,
    )
    class_sizes = predict_class_sizes(self_sim, th=th, vote_type=vote_type)
    if return_class_sizes:
        return class_sizes

    is_few_shot = predict_few_shot_class_samples(class_sizes, percentile=percentile)
    sample_indices = torch.where(is_few_shot == 1)[0]
    return X[sample_indices], sample_indices


def compute_orthogonal_distances(points: torch.Tensor, line_points) -> torch.Tensor:
    points_np = points.numpy()
    (x1, y1), (x2, y2) = line_points[0].numpy(), line_points[1].numpy()
    dx, dy = x2 - x1, y2 - y1
    if dx == 0:
        return torch.FloatTensor([abs(x - x1) for x, _ in points_np])

    slope = dy / dx
    intercept = y1 - slope * x1
    perp_slope = -1 / slope

    distances = []
    for x, y in points_np:
        perp_intercept = y - perp_slope * x
        x_intersect = (perp_intercept - intercept) / (slope - perp_slope)
        y_intersect = slope * x_intersect + intercept
        distances.append(np.sqrt((x_intersect - x) ** 2 + (y_intersect - y) ** 2))

    return torch.FloatTensor(distances)


def predict_adaptive_class_sizes(self_sim: torch.Tensor, th_type: str = 'double_max_step', vote_type: str = 'none'):
    ths = _compute_ths(self_sim, th_type)
    mask = self_sim >= ths
    count_map = mask * mask.sum(dim=1, keepdim=True)
    return _vote(count_map, vote_type)


def _vote(count_map: torch.Tensor, vote_type):
    if vote_type == 'none':
        return _vote_none(count_map)
    if vote_type == 'mode':
        return _vote_mode(count_map)
    if vote_type == 'mean':
        return _vote_mean(count_map)
    raise ValueError(f'unsupported adaptive vote_type: {vote_type}')


def _vote_mean(count_map: torch.Tensor):
    class_sizes = count_map.sum(dim=0) / torch.count_nonzero(count_map, dim=0)
    return class_sizes.to(torch.float)


def _vote_none(count_map: torch.Tensor):
    class_sizes = count_map.float().mean(dim=1)
    return class_sizes.to(torch.float)


def _vote_mode(count_map: torch.Tensor):
    count_map = count_map.numpy()
    count_map_naned = np.where(count_map == 0, np.nan, count_map)
    class_sizes = mode(count_map_naned, axis=0, nan_policy='omit').mode
    return torch.from_numpy(class_sizes).to(torch.float)


def _compute_ths(self_sim: torch.Tensor, th_type: str) -> torch.Tensor:
    self_sim = self_sim.numpy()

    if th_type == 'max_step':
        compute_th_fn = _compute_th_max_step
    elif th_type == 'double_max_step':
        compute_th_fn = _compute_th_double_max_step
    elif th_type == 'max_step_min_num_neighbors':
        compute_th_fn = _compute_th_max_step_min_num_neighbors
    elif th_type == 'double_min_bin_count':
        compute_th_fn = _compute_th_double_min_bin_count
    elif th_type == 'min_kde':
        compute_th_fn = _compute_th_min_kde
    elif th_type == 'double_min_kde':
        compute_th_fn = _compute_th_double_min_kde
    elif th_type == 'half_min':
        compute_th_fn = _compute_th_half_min
    elif th_type == 'ruled_max_step':
        compute_th_fn = _compute_th_ruled_max_step
    elif th_type == 'trim_min':
        compute_th_fn = _compute_th_trim_min
    elif th_type == 'truncate_min':
        compute_th_fn = _compute_th_truncate_min
    else:
        raise ValueError(f'unsupported adaptive threshold_type: {th_type}')

    def compute_one(index):
        return compute_th_fn(self_sim[index])

    ths = Parallel(n_jobs=-1)(delayed(compute_one)(index) for index in range(len(self_sim)))
    return torch.FloatTensor(ths)[:, None]


def _compute_th_max_step(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    diff = np.diff(scores_sorted)
    idx_max_step = diff.argmax()
    return (scores_sorted[idx_max_step + 1] + scores_sorted[idx_max_step]) / 2


def _compute_th_double_max_step(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    diffs = np.diff(scores_sorted)
    idx_max_step = _double_criterion(diffs)
    return (scores_sorted[idx_max_step] + scores_sorted[idx_max_step + 1]) / 2


def _double_criterion(crits: np.ndarray):
    next_crits = np.empty_like(crits)
    for index in range(len(next_crits) - 1):
        next_crits[index] = np.max(crits[index + 1:])
    next_crits[-1] = crits[-1]
    final_crits = crits / np.maximum(next_crits, 1e-7)
    return final_crits.argmax()


def _compute_th_double_min_bin_count(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    bin_counts = scores_to_bin_counts(scores_sorted)[:-1]
    idx = _double_criterion(bin_counts.max() - bin_counts)
    return (scores_sorted[idx] + scores_sorted[idx + 1]) / 2


def _compute_th_min_kde(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    kde_log_density = scores_to_kde(scores_sorted)[:-1]
    idx = kde_log_density.argmin()
    return (scores_sorted[idx] + scores_sorted[idx + 1]) / 2


def _compute_th_double_min_kde(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    kde_log_density = scores_to_kde(scores_sorted)[:-1]
    idx = _double_criterion(kde_log_density.max() - kde_log_density)
    return (scores_sorted[idx] + scores_sorted[idx + 1]) / 2


def _compute_th_half_min(scores: np.ndarray):
    minimum = scores.min()
    return np.cos(np.arccos(minimum) / 2)


def _compute_th_trim_min(scores: np.ndarray):
    scores = _sort(scores, descending=False)
    minimum = scores.min()
    th_half = np.cos(np.arccos(minimum) / 2)
    scores_half = scores[scores > th_half]
    p = 0.85
    scores_trim = scores_half[-int(len(scores_half) * p)]
    return scores_trim


def _compute_th_truncate_min(scores: np.ndarray):
    scores = _sort(scores, descending=False)
    minimum = scores.min()
    th_half = np.cos(np.arccos(minimum) / 2)
    scores_half = scores[scores > th_half]
    p = 0.5
    scores_trim = scores_half[-int(len(scores_half) * p)]
    return scores_trim


def _compute_th_ruled_max_step(scores: np.ndarray):
    minimum = scores.min()
    th_quarter = np.cos(np.arccos(minimum) / 8)
    th_half = np.cos(np.arccos(minimum) / 2)
    th_double_max_step = _compute_th_double_max_step(scores)
    if th_double_max_step >= th_quarter:
        return th_half
    return th_double_max_step


def _compute_th_max_step_min_num_neighbors(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=True)
    diff = -np.diff(scores_sorted)
    crit1 = diff
    crit2 = -np.log(np.arange(len(diff)) + 1) + math.log(len(diff))
    crit = crit1 * crit2
    idx_max_step = crit.argmax()
    return (scores_sorted[idx_max_step] + scores_sorted[idx_max_step + 1]) / 2


def _sort(scores: np.ndarray, descending: bool = False, quantize: bool = True) -> np.ndarray:
    dtype = scores.dtype
    if quantize:
        n = len(scores)
        scores = (n * scores).astype(np.int_)
    else:
        n = 1

    scores = np.sort(scores).astype(dtype) / n
    if descending:
        scores = scores[::-1]
    return scores


def adaptively_sample_few_shot(
    X: torch.Tensor,
    th_type: str = 'double_max_step',
    vote_type: str = 'none',
    percentile: float = 0.15,
    return_class_sizes: bool = False,
    partition_guard: Optional[str] = None,
    return_partition_diagnostics: bool = False,
):
    self_sim = compute_self_sim(X)
    class_sizes = predict_adaptive_class_sizes(self_sim, th_type, vote_type).squeeze()
    if return_class_sizes:
        if return_partition_diagnostics:
            raise ValueError('partition diagnostics require candidate selection')
        return class_sizes

    is_few_shot, diagnostics = predict_few_shot_class_samples(
        class_sizes,
        percentile=percentile,
        partition_guard=partition_guard,
        return_partition_diagnostics=True,
    )
    sample_indices = torch.where(is_few_shot == 1)[0]
    if return_partition_diagnostics:
        return X[sample_indices], sample_indices, diagnostics
    return X[sample_indices], sample_indices


def scores_to_bin_counts(scores):
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    n = len(scores)
    bin_width = 2 * iqr / (n ** (1 / 3))
    data_range = np.max(scores) - np.min(scores)
    num_bins = int(np.round(data_range / bin_width))

    hist, bin_edges = np.histogram(scores, bins=num_bins)
    bin_counts = np.zeros_like(scores)
    for index in range(num_bins):
        if index < num_bins - 1:
            in_bin = (scores >= bin_edges[index]) & (scores < bin_edges[index + 1])
        else:
            in_bin = scores >= bin_edges[index]
        bin_counts[in_bin] = hist[index]
    return bin_counts


def scores_to_kde(scores):
    scores = np.array(scores).reshape(-1, 1)
    kde = KernelDensity(kernel='tophat', bandwidth='silverman').fit(scores)
    return kde.score_samples(scores)


class TailSampler:
    def __init__(
        self,
        threshold_type: str = 'symmin',
        vote_type: str = 'mean',
        percentile: float = 0.15,
        num_bootstrapping: int = 1,
        subsampling_ratio: float = 1.0,
    ):
        self.threshold_type = threshold_type
        self.vote_type = vote_type
        self.percentile = float(percentile)
        self.num_bootstrapping = int(num_bootstrapping)
        self.subsampling_ratio = float(subsampling_ratio)

    def run(self, features: torch.Tensor, feature_map_shape: Optional[torch.Tensor] = None, return_class_sizes: bool = False):
        tail_samples, tail_indices = sample_few_shot(
            X=features,
            fea_map_shape=feature_map_shape,
            th_type=self.threshold_type,
            vote_type=self.vote_type,
            num_bootstrapping=self.num_bootstrapping,
            subsampling_ratio=self.subsampling_ratio,
            percentile=self.percentile,
        )
        if return_class_sizes:
            class_sizes_pred = sample_few_shot(
                X=features,
                fea_map_shape=feature_map_shape,
                th_type=self.threshold_type,
                vote_type=self.vote_type,
                num_bootstrapping=self.num_bootstrapping,
                subsampling_ratio=self.subsampling_ratio,
                percentile=self.percentile,
                return_class_sizes=True,
            )
            return tail_samples, tail_indices, class_sizes_pred
        return tail_samples, tail_indices


class AdaptiveTailSampler:
    def __init__(
        self,
        threshold_type: str = 'double_max_step',
        vote_type: str = 'none',
        percentile: float = 0.15,
        partition_guard: Optional[str] = None,
    ):
        self.threshold_type = threshold_type
        self.vote_type = vote_type
        self.percentile = float(percentile)
        self.partition_guard = partition_guard
        self.last_partition_diagnostics = None

    def run(self, features: torch.Tensor, feature_map_shape: Optional[torch.Tensor] = None, return_class_sizes: bool = False):
        if feature_map_shape is not None:
            raise ValueError('adaptive tail sampler image-level port does not support feature_map_shape')

        tail_samples, tail_indices, partition_diagnostics = adaptively_sample_few_shot(
            X=features,
            th_type=self.threshold_type,
            vote_type=self.vote_type,
            percentile=self.percentile,
            partition_guard=self.partition_guard,
            return_partition_diagnostics=True,
        )
        self.last_partition_diagnostics = partition_diagnostics
        if return_class_sizes:
            class_sizes_pred = adaptively_sample_few_shot(
                X=features,
                th_type=self.threshold_type,
                vote_type=self.vote_type,
                percentile=self.percentile,
                return_class_sizes=True,
            )
            return tail_samples, tail_indices, class_sizes_pred
        return tail_samples, tail_indices


def build_tail_sampler(
    sampler_type: str = 'adaptive',
    threshold_type: Optional[str] = None,
    vote_type: Optional[str] = None,
    percentile: float = 0.15,
    plateau_gap_guard: bool = False,
):
    if sampler_type == 'tail':
        return TailSampler(
            threshold_type='symmin' if threshold_type is None else threshold_type,
            vote_type='mean' if vote_type is None else vote_type,
            percentile=percentile,
        ), 'TailSampler'

    if sampler_type == 'adaptive':
        return AdaptiveTailSampler(
            threshold_type='double_max_step' if threshold_type is None else threshold_type,
            vote_type='none' if vote_type is None else vote_type,
            percentile=percentile,
        ), 'AdaptiveTailSampler'

    if sampler_type == 'adaptive_trim_mode':
        effective_threshold_type = 'trim_min' if threshold_type is None else threshold_type
        effective_vote_type = 'mode' if vote_type is None else vote_type
        if plateau_gap_guard and (
            effective_threshold_type != 'trim_min' or effective_vote_type != 'mode'
        ):
            raise ValueError('plateau-gap guard requires adaptive_trim_mode with trim_min and mode')
        return AdaptiveTailSampler(
            threshold_type=effective_threshold_type,
            vote_type=effective_vote_type,
            percentile=percentile,
            partition_guard=PLATEAU_GAP_GUARD_V1 if plateau_gap_guard else None,
        ), 'AdaptiveTailSampler(trim_min,mode)'

    raise ValueError(f'unsupported sampler_type: {sampler_type}')
