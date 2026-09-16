"""Tail reconciliation through relative group dominance."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


GEOMETRY_COLUMNS = [
    'group_id',
    'group_size',
    'valid_group',
    'distance_median',
    'distance_mad',
    'distance_min',
    'distance_max',
]


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), dim=-1)


def _embedding_lookup(embeddings_payload: Dict):
    sample_indices = [int(value) for value in embeddings_payload['sample_idx']]
    embeddings = _normalize_rows(embeddings_payload['embeddings'].float())
    return {sample_idx: embeddings[index] for index, sample_idx in enumerate(sample_indices)}


def _linear_sse(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    design = np.stack([x, np.ones_like(x)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return float(np.sum(residual ** 2))


def _bic(sse: float, n: int, q: int) -> float:
    n = max(1, int(n))
    return float(n * np.log(float(sse) / n + 1e-12) + int(q) * np.log(n))


def build_clean_head_geometry(h_clean_df: pd.DataFrame, grouping_embeddings_payload: Dict, args) -> Tuple[Dict, pd.DataFrame]:
    clean_df = h_clean_df.copy()
    if len(clean_df) == 0:
        return {'groups': {}, 'embedding_source': grouping_embeddings_payload.get('embedding_source')}, pd.DataFrame(columns=GEOMETRY_COLUMNS)
    clean_df['sample_idx'] = clean_df['sample_idx'].astype(int)
    clean_df['group_id'] = clean_df['group_id'].astype(int)
    embedding_map = _embedding_lookup(grouping_embeddings_payload)
    min_group_size = int(getattr(args, 'tg_min_clean_group_size', 3))

    groups = {}
    rows = []
    for group_id, group_df in clean_df.groupby('group_id', sort=True):
        vectors = []
        sample_indices = []
        for sample_idx in group_df['sample_idx'].astype(int).tolist():
            vector = embedding_map.get(int(sample_idx))
            if vector is None:
                continue
            vectors.append(vector)
            sample_indices.append(int(sample_idx))
        if len(vectors) == 0:
            continue
        matrix = torch.stack(vectors, dim=0)
        centroid = _normalize_rows(matrix.mean(dim=0, keepdim=True))[0]
        distances = (1.0 - torch.clamp(matrix @ centroid, min=-1.0, max=1.0)).cpu().numpy().astype(float)
        valid_group = len(sample_indices) >= min_group_size
        group_id = int(group_id)
        groups[group_id] = {
            'group_id': group_id,
            'centroid': centroid.cpu(),
            'distances': distances,
            'sample_indices': sample_indices,
            'valid_group': bool(valid_group),
            'group_size': int(len(sample_indices)),
        }
        rows.append({
            'group_id': group_id,
            'group_size': int(len(sample_indices)),
            'valid_group': bool(valid_group),
            'distance_median': float(np.median(distances)),
            'distance_mad': float(np.median(np.abs(distances - np.median(distances)))),
            'distance_min': float(np.min(distances)),
            'distance_max': float(np.max(distances)),
        })

    geometry = {
        'groups': groups,
        'embedding_source': grouping_embeddings_payload.get('embedding_source'),
        'min_clean_group_size': int(min_group_size),
        'num_groups': int(len(groups)),
        'num_valid_groups': int(sum(1 for entry in groups.values() if entry['valid_group'])),
    }
    return geometry, pd.DataFrame(rows, columns=GEOMETRY_COLUMNS)


RGD_RAW_DISTANCE_COLUMNS = [
    'sample_idx',
    'sample_key',
    'group_id',
    'distance',
    'conformity',
]


RGD_SCORE_COLUMNS = [
    'best_group_id',
    'best_distance',
    'second_group_id',
    'second_distance',
    'other_group_median_distance',
    'relative_group_dominance',
    'rgd_threshold',
    'rgd_status',
]


def _is_finite_nonzero_vector(vector: torch.Tensor) -> bool:
    return bool(
        torch.isfinite(vector).all().item()
        and torch.linalg.vector_norm(vector).item() > 1e-12
    )


def _new_rgd_score_row(row: pd.Series, status: str):
    score_row = row.to_dict()
    score_row.update({
        'best_group_id': -1,
        'best_distance': np.nan,
        'second_group_id': -1,
        'second_distance': np.nan,
        'other_group_median_distance': np.nan,
        'relative_group_dominance': np.nan,
        'rgd_threshold': np.nan,
        'rgd_status': status,
    })
    return score_row


def compute_tail_relative_group_dominance(tail_df: pd.DataFrame,
                                          geometry: Dict,
                                          grouping_embeddings_payload: Dict):
    tail_df = tail_df.copy().reset_index(drop=True)
    embedding_map = _embedding_lookup(grouping_embeddings_payload)
    valid_groups = sorted(
        (
            (int(group_id), entry)
            for group_id, entry in geometry.get('groups', {}).items()
            if (
                bool(entry.get('valid_group', False))
                and _is_finite_nonzero_vector(entry['centroid'])
            )
        ),
        key=lambda item: item[0],
    )
    raw_distance_rows = []
    score_rows = []

    for _, row in tail_df.iterrows():
        sample_idx = int(row['sample_idx'])
        sample_key = int(row.get('sample_key', sample_idx))
        if len(valid_groups) < 2:
            score_rows.append(_new_rgd_score_row(row, 'open_insufficient_valid_groups'))
            continue

        vector = embedding_map.get(sample_idx)
        if vector is None:
            score_rows.append(_new_rgd_score_row(row, 'open_missing_embedding'))
            continue
        if not _is_finite_nonzero_vector(vector):
            score_rows.append(_new_rgd_score_row(row, 'open_invalid_embedding'))
            continue

        group_distances = []
        has_invalid_distance = False
        for group_id, entry in valid_groups:
            centroid = entry['centroid'].to(vector.device)
            distance = float(1.0 - torch.clamp(vector @ centroid, min=-1.0, max=1.0).item())
            raw_distance_rows.append({
                'sample_idx': sample_idx,
                'sample_key': sample_key,
                'group_id': group_id,
                'distance': distance,
                'conformity': np.nan,
            })
            if not np.isfinite(distance):
                has_invalid_distance = True
            group_distances.append((group_id, distance))

        if has_invalid_distance:
            score_rows.append(_new_rgd_score_row(row, 'open_invalid_distance'))
            continue

        group_distances.sort(key=lambda item: (item[1], item[0]))
        best_group_id, best_distance = group_distances[0]
        second_group_id, second_distance = group_distances[1]
        other_distances = np.asarray(
            [distance for _, distance in group_distances[1:]],
            dtype=np.float64,
        )
        other_median = float(np.median(other_distances))
        if not np.isfinite(other_median) or other_median <= 1e-12:
            score_rows.append(_new_rgd_score_row(row, 'open_invalid_rgd'))
            continue

        dominance = float(1.0 - float(best_distance) / other_median)
        if not np.isfinite(dominance):
            score_rows.append(_new_rgd_score_row(row, 'open_invalid_rgd'))
            continue

        score_row = _new_rgd_score_row(row, 'pending_rgd_split')
        score_row.update({
            'best_group_id': int(best_group_id),
            'best_distance': float(best_distance),
            'second_group_id': int(second_group_id),
            'second_distance': float(second_distance),
            'other_group_median_distance': other_median,
            'relative_group_dominance': dominance,
        })
        score_rows.append(score_row)

    raw_distances_df = pd.DataFrame(raw_distance_rows, columns=RGD_RAW_DISTANCE_COLUMNS)
    scores_df = pd.DataFrame(score_rows)
    if len(scores_df) == 0:
        scores_df = tail_df.copy()
        for column in RGD_SCORE_COLUMNS:
            scores_df[column] = []
    return raw_distances_df, scores_df, int(len(valid_groups))


RGD_BIC_CANDIDATE_COLUMNS = [
    'split_index',
    'left_count',
    'right_count',
    'left_sse',
    'right_sse',
    'sse_2',
    'bic_2',
    'is_selected',
]


def _empty_rgd_bic_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=RGD_BIC_CANDIDATE_COLUMNS)


def _rgd_fallback(scores_df: pd.DataFrame,
                  scoreable_index: pd.Index,
                  summary: Dict,
                  bic_candidates_df: Optional[pd.DataFrame] = None):
    output_df = scores_df.copy()
    if len(scoreable_index) > 0:
        output_df.loc[scoreable_index, 'rgd_status'] = 'open_no_valid_rgd_split'
    open_df = output_df.copy().reset_index(drop=True)
    attached_df = output_df.iloc[0:0].copy().reset_index(drop=True)
    summary.update({
        'split_valid': False,
        'num_open': int(len(open_df)),
        'num_head_affiliated': 0,
    })
    return output_df, open_df, attached_df, summary, (
        _empty_rgd_bic_candidates() if bic_candidates_df is None else bic_candidates_df
    )


def _apply_rgd_split(scores_df: pd.DataFrame,
                     scoreable_mask: pd.Series,
                     scoreable_index: pd.Index,
                     threshold: float,
                     summary: Dict,
                     bic_candidates_df: pd.DataFrame):
    output_df = scores_df.copy()
    output_df.loc[scoreable_index, 'rgd_threshold'] = threshold
    attached_mask = scoreable_mask & (output_df['relative_group_dominance'] > threshold)
    output_df.loc[scoreable_mask & ~attached_mask, 'rgd_status'] = 'open'
    output_df.loc[attached_mask, 'rgd_status'] = 'head_affiliated'
    open_df = output_df.loc[output_df['rgd_status'] != 'head_affiliated'].copy().reset_index(drop=True)
    attached_df = output_df.loc[output_df['rgd_status'] == 'head_affiliated'].copy().reset_index(drop=True)
    summary.update({
        'split_valid': True,
        'num_open': int(len(open_df)),
        'num_head_affiliated': int(len(attached_df)),
    })
    return output_df, open_df, attached_df, summary, bic_candidates_df


def _rgd_candidate_row(split_index: int,
                       x: np.ndarray,
                       values: np.ndarray):
    lower_count = int(split_index + 1)
    right_count = int(len(values) - lower_count)
    left_sse = _linear_sse(x[:lower_count], values[:lower_count])
    right_sse = _linear_sse(x[lower_count:], values[lower_count:])
    sse_2 = float(left_sse + right_sse)
    return {
        'split_index': int(split_index),
        'left_count': lower_count,
        'right_count': right_count,
        'left_sse': float(left_sse),
        'right_sse': float(right_sse),
        'sse_2': sse_2,
        'bic_2': _bic(sse_2, len(values), q=4),
        'is_selected': False,
    }


def split_tail_by_relative_group_dominance(rgd_scores_df: pd.DataFrame,
                                            num_valid_groups: int,
                                            args):
    scores_df = rgd_scores_df.copy().reset_index(drop=True)
    split_mode = getattr(args, 'tg_rgd_split_mode', 'segmented_bic')
    if split_mode not in {'segmented_bic', 'largest_positive_gap'}:
        raise ValueError('unknown TailGuard RGD split mode: {}'.format(split_mode))
    configured_min_segment = int(getattr(args, 'tg_elbow_min_segment', 3))
    min_segment = max(3, configured_min_segment) if split_mode == 'segmented_bic' else max(1, configured_min_segment)
    base_summary = {
        'mode': 'rgd',
        'split_mode': split_mode,
        'candidate_selection': 'all_legal_breakpoints' if split_mode == 'segmented_bic' else 'largest_positive_gap',
        'num_tail': int(len(scores_df)),
        'num_valid_groups': int(num_valid_groups),
        'min_segment': int(min_segment),
        'num_scoreable_rgd': 0,
        'num_unique_scores': 0,
        'num_legal_breakpoints': 0,
        'max_gap': None,
        'split_index': None,
        'threshold': None,
        'sse_1': None,
        'sse_2': None,
        'bic_1': None,
        'bic_2': None,
    }
    if len(scores_df) == 0:
        return scores_df, scores_df.copy(), scores_df.copy(), {
            **base_summary,
            'status': 'empty_tail',
            'split_valid': False,
            'num_open': 0,
            'num_head_affiliated': 0,
        }, _empty_rgd_bic_candidates()
    if num_valid_groups < 2:
        return _rgd_fallback(
            scores_df,
            scores_df.index[scores_df['rgd_status'] == 'pending_rgd_split'],
            {**base_summary, 'status': 'fallback_insufficient_valid_groups'},
        )

    scoreable_mask = (
        (scores_df['rgd_status'] == 'pending_rgd_split')
        & np.isfinite(scores_df['relative_group_dominance'].to_numpy(dtype=float))
    )
    scoreable_df = scores_df.loc[scoreable_mask].sort_values(
        ['relative_group_dominance', 'sample_idx'],
        ascending=True,
        kind='mergesort',
    )
    scoreable_index = scoreable_df.index
    values = scoreable_df['relative_group_dominance'].to_numpy(dtype=np.float64)
    x = np.arange(len(scoreable_df), dtype=float)
    base_summary.update({
        'num_scoreable_rgd': int(len(scoreable_df)),
        'num_unique_scores': int(np.unique(values).size),
    })
    minimum_scoreable = max(2 * min_segment, 4) if split_mode == 'largest_positive_gap' else 2 * min_segment
    if len(scoreable_df) < minimum_scoreable:
        return _rgd_fallback(scores_df, scoreable_index, {
            **base_summary,
            'status': 'fallback_insufficient_scoreable_tail',
        })

    if split_mode == 'segmented_bic':
        sse_1 = _linear_sse(x, values)
        bic_1 = _bic(sse_1, len(scoreable_df), q=2)
        base_summary.update({'sse_1': float(sse_1), 'bic_1': float(bic_1)})
        candidate_rows = [
            _rgd_candidate_row(split_index, x, values)
            for split_index in range(min_segment - 1, len(scoreable_df) - min_segment)
            if values[split_index] < values[split_index + 1]
        ]
        candidates_df = pd.DataFrame(candidate_rows, columns=RGD_BIC_CANDIDATE_COLUMNS)
        base_summary['num_legal_breakpoints'] = int(len(candidates_df))
        if len(candidates_df) == 0:
            return _rgd_fallback(scores_df, scoreable_index, {
                **base_summary,
                'status': 'fallback_no_legal_breakpoint',
            }, candidates_df)
        best_position = int(candidates_df['bic_2'].idxmin())
        candidates_df.loc[best_position, 'is_selected'] = True
        best = candidates_df.loc[best_position]
        base_summary.update({
            'split_index': int(best['split_index']),
            'sse_2': float(best['sse_2']),
            'bic_2': float(best['bic_2']),
        })
        if not (float(best['bic_2']) < bic_1):
            return _rgd_fallback(scores_df, scoreable_index, {
                **base_summary,
                'status': 'fallback_no_bic_gain',
            }, candidates_df)
        split_index = int(best['split_index'])
        threshold = float((values[split_index] + values[split_index + 1]) / 2.0)
        summary = {
            **base_summary,
            'status': 'segmented_bic_split',
            'threshold': threshold,
        }
        return _apply_rgd_split(
            scores_df,
            scoreable_mask,
            scoreable_index,
            threshold,
            summary,
            candidates_df,
        )

    gaps = np.diff(values)
    positive_gap_indices = np.flatnonzero(gaps > 0.0)
    if len(positive_gap_indices) == 0:
        return _rgd_fallback(scores_df, scoreable_index, {
            **base_summary,
            'status': 'fallback_no_positive_gap',
        })
    split_index = int(positive_gap_indices[np.argmax(gaps[positive_gap_indices])])
    candidate = _rgd_candidate_row(split_index, x, values)
    candidates_df = pd.DataFrame([candidate], columns=RGD_BIC_CANDIDATE_COLUMNS)
    base_summary.update({
        'max_gap': float(gaps[split_index]),
        'split_index': split_index,
    })
    if candidate['left_count'] < min_segment or candidate['right_count'] < min_segment:
        return _rgd_fallback(scores_df, scoreable_index, {
            **base_summary,
            'status': 'fallback_largest_gap_violates_min_segment',
        }, candidates_df)
    candidates_df.loc[0, 'is_selected'] = True
    sse_1 = _linear_sse(x, values)
    bic_1 = _bic(sse_1, len(scoreable_df), q=2)
    base_summary.update({
        'num_legal_breakpoints': 1,
        'sse_1': float(sse_1),
        'sse_2': float(candidate['sse_2']),
        'bic_1': float(bic_1),
        'bic_2': float(candidate['bic_2']),
    })
    if not (float(candidate['bic_2']) < bic_1):
        return _rgd_fallback(scores_df, scoreable_index, {
            **base_summary,
            'status': 'fallback_no_bic_gain',
        }, candidates_df)
    threshold = float((values[split_index] + values[split_index + 1]) / 2.0)
    summary = {
        **base_summary,
        'status': 'largest_gap_bic_split',
        'threshold': threshold,
    }
    return _apply_rgd_split(
        scores_df,
        scoreable_mask,
        scoreable_index,
        threshold,
        summary,
        candidates_df,
    )


def build_tail_attachment_plan(
    tail_df: pd.DataFrame,
    h_clean_df: pd.DataFrame,
    grouping_embeddings_payload: Dict,
    args,
    membership_mode: str,
):
    """Reconcile protected tail candidates with the purified head structure."""
    if membership_mode != "rgd":
        raise ValueError("the TailGuard mainline requires membership_mode='rgd'")
    geometry, _ = build_clean_head_geometry(
        h_clean_df,
        grouping_embeddings_payload,
        args,
    )
    raw_distances_df, scores_df, num_valid_groups = compute_tail_relative_group_dominance(
        tail_df,
        geometry,
        grouping_embeddings_payload,
    )
    (
        scores_df,
        tail_open_df,
        tail_attached_df,
        rgd_summary,
        rgd_bic_candidates_df,
    ) = split_tail_by_relative_group_dominance(scores_df, num_valid_groups, args)
    attachment_summary = dict(rgd_summary)
    attachment_summary.update(
        {
            "elbow_valid": bool(rgd_summary["split_valid"]),
            "break_index": rgd_summary["split_index"],
            "num_tail_open": int(rgd_summary["num_open"]),
            "num_tail_attached": int(rgd_summary["num_head_affiliated"]),
        }
    )
    return {
        "geometry": geometry,
        "conformity_df": pd.DataFrame(
            columns=["sample_idx", "sample_key", "group_id", "distance", "conformity"]
        ),
        "attachment_scores_df": scores_df,
        "tail_open_df": tail_open_df,
        "tail_attached_df": tail_attached_df,
        "attachment_summary": attachment_summary,
        "membership_scores_df": None,
        "membership_calibration_df": None,
        "membership_summary": None,
        "rgd_distances_df": raw_distances_df,
        "rgd_scores_df": scores_df,
        "rgd_split_summary": rgd_summary,
        "rgd_bic_candidates_df": rgd_bic_candidates_df,
    }

