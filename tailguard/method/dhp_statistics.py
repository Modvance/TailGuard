import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


DEFAULT_BOOTSTRAP_B = 20


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
        return None
    return float(value)


def robust_normalize_group_scores(scores, eps=1e-6):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError('cannot normalize empty scores')
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) + float(eps)
    z_scores = (scores - median) / mad
    return z_scores.astype(float), {'score_median': median, 'score_mad': mad}


def fit_group_gmms(z_scores, eps=1e-6, random_state=0):
    z_scores = np.asarray(z_scores, dtype=float).reshape(-1)
    if z_scores.size == 0:
        raise ValueError('cannot fit GMM on empty scores')

    x = z_scores.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, covariance_type='full', reg_covar=eps, random_state=random_state)
    gmm1.fit(x)
    bic_1 = float(gmm1.bic(x))

    if z_scores.size < 2 or np.allclose(z_scores, z_scores[0]):
        mean_value = float(np.mean(z_scores))
        sigma_value = float(np.std(z_scores) + eps)
        posterior_high = np.full(z_scores.shape[0], 0.5, dtype=float)
        return {
            'bic_1': bic_1,
            'bic_2': bic_1,
            'delta_bic': 0.0,
            'mu_low': mean_value,
            'mu_high': mean_value,
            'sigma_low': sigma_value,
            'sigma_high': sigma_value,
            'pi_low': 0.5,
            'pi_high': 0.5,
            'posterior_high': posterior_high,
        }

    try:
        gmm2 = GaussianMixture(n_components=2, covariance_type='full', reg_covar=eps, random_state=random_state)
        gmm2.fit(x)
        bic_2 = float(gmm2.bic(x))
        means = gmm2.means_.reshape(-1)
        variances = gmm2.covariances_.reshape(2, -1)[:, 0]
        sigmas = np.sqrt(np.maximum(variances, eps))
        weights = gmm2.weights_.reshape(-1)
        order = np.argsort(means)
        low_idx, high_idx = int(order[0]), int(order[1])
        posterior_high = gmm2.predict_proba(x)[:, high_idx]
        return {
            'bic_1': bic_1,
            'bic_2': bic_2,
            'delta_bic': float(bic_1 - bic_2),
            'mu_low': float(means[low_idx]),
            'mu_high': float(means[high_idx]),
            'sigma_low': float(sigmas[low_idx]),
            'sigma_high': float(sigmas[high_idx]),
            'pi_low': float(weights[low_idx]),
            'pi_high': float(weights[high_idx]),
            'posterior_high': posterior_high.astype(float),
        }
    except Exception:
        mean_value = float(np.mean(z_scores))
        sigma_value = float(np.std(z_scores) + eps)
        posterior_high = np.full(z_scores.shape[0], 0.5, dtype=float)
        return {
            'bic_1': bic_1,
            'bic_2': bic_1,
            'delta_bic': 0.0,
            'mu_low': mean_value,
            'mu_high': mean_value,
            'sigma_low': sigma_value,
            'sigma_high': sigma_value,
            'pi_low': 0.5,
            'pi_high': 0.5,
            'posterior_high': posterior_high,
        }


def compute_group_sep_conf(gmm_stats, eps=1e-6):
    sep = float((gmm_stats['mu_high'] - gmm_stats['mu_low']) / max(gmm_stats['sigma_low'] + gmm_stats['sigma_high'], eps))
    posterior_high = np.asarray(gmm_stats['posterior_high'], dtype=float)
    posterior = np.stack([1.0 - posterior_high, posterior_high], axis=1)
    entropy = -np.sum(posterior * np.log(posterior + float(eps)), axis=1)
    H = float(np.mean(entropy) / math.log(2.0)) if posterior.shape[0] > 0 else 1.0
    conf = float(np.clip(1.0 - H, 0.0, 1.0))
    return sep, conf


def evaluate_group_noise_gate(group_size, delta_bic, sep, conf, pi_high, args):
    if int(group_size) < int(args.gbps_min_group_size):
        return False, 0.0, 'small_group'
    if getattr(args, 'gbps_gate_mode', 'hard') != 'hard':
        raise ValueError(f"unsupported gbps_gate_mode: {args.gbps_gate_mode}")

    is_active = (
        float(delta_bic) > float(args.gbps_tau_bic)
        and float(sep) > float(args.gbps_tau_sep)
        and float(conf) > float(args.gbps_tau_conf)
        and float(args.gbps_pi_min) <= float(pi_high) <= float(args.gbps_pi_max)
    )
    return bool(is_active), (1.0 if is_active else 0.0), ('active' if is_active else 'gate_rejected')


def compute_group_metrics_for_scores(group_df, args, random_state=0):
    group_df = group_df.copy().reset_index(drop=True)
    group_id = int(group_df['group_id'].iloc[0])
    group_size = int(len(group_df))

    if group_size < int(args.gbps_min_group_size):
        small_df = group_df.copy()
        small_df['z_score'] = np.nan
        small_df['p_high'] = np.nan
        small_df['q_g'] = 0.0
        small_df['is_active_group'] = False
        metrics = {
            'group_id': group_id,
            'group_size': group_size,
            'valid_group': False,
            'bic_1': None,
            'bic_2': None,
            'delta_bic': None,
            'mu_low': None,
            'mu_high': None,
            'sigma_low': None,
            'sigma_high': None,
            'pi_low': None,
            'pi_high': None,
            'sep': None,
            'conf': None,
            'q_g': 0.0,
            'U_g': 0.0,
            'is_active_group': False,
            'skip_reason': 'small_group',
            'score_median': None,
            'score_mad': None,
        }
        return metrics, small_df

    z_scores, robust_stats = robust_normalize_group_scores(group_df['image_score'].to_numpy(dtype=float))
    gmm_stats = fit_group_gmms(z_scores, eps=1e-6, random_state=random_state)
    sep, conf = compute_group_sep_conf(gmm_stats)
    is_active_group, q_g, skip_reason = evaluate_group_noise_gate(
        group_size,
        gmm_stats['delta_bic'],
        sep,
        conf,
        gmm_stats['pi_high'],
        args,
    )
    U_g = float(q_g) * float(sep) * float(conf)

    sample_df = group_df.copy()
    sample_df['z_score'] = z_scores.astype(float)
    sample_df['p_high'] = np.asarray(gmm_stats['posterior_high'], dtype=float)
    sample_df['q_g'] = float(q_g)
    sample_df['is_active_group'] = bool(is_active_group)

    metrics = {
        'group_id': group_id,
        'group_size': group_size,
        'valid_group': True,
        'bic_1': _safe_float(gmm_stats['bic_1']),
        'bic_2': _safe_float(gmm_stats['bic_2']),
        'delta_bic': _safe_float(gmm_stats['delta_bic']),
        'mu_low': _safe_float(gmm_stats['mu_low']),
        'mu_high': _safe_float(gmm_stats['mu_high']),
        'sigma_low': _safe_float(gmm_stats['sigma_low']),
        'sigma_high': _safe_float(gmm_stats['sigma_high']),
        'pi_low': _safe_float(gmm_stats['pi_low']),
        'pi_high': _safe_float(gmm_stats['pi_high']),
        'sep': _safe_float(sep),
        'conf': _safe_float(conf),
        'q_g': float(q_g),
        'U_g': _safe_float(U_g),
        'is_active_group': bool(is_active_group),
        'skip_reason': skip_reason,
        'score_median': _safe_float(robust_stats['score_median']),
        'score_mad': _safe_float(robust_stats['score_mad']),
    }
    return metrics, sample_df


def _aggregate_group_metrics(group_metrics_df, sample_group_df):
    valid_df = group_metrics_df[group_metrics_df['valid_group'] == True].copy()
    if len(valid_df) == 0:
        return {
            'gbps_U': 0.0,
            'gbps_noise_evidence': 0.0,
            'gbps_active_group_count': 0,
            'gbps_valid_group_count': 0,
            'gbps_active_sample_ratio': 0.0,
        }

    active_group_ids = set(valid_df.loc[valid_df['q_g'] > 0, 'group_id'].astype(int).tolist())
    active_sample_ratio = 0.0
    if len(sample_group_df) > 0:
        active_sample_ratio = float(sample_group_df['group_id'].isin(active_group_ids).mean())

    return {
        'gbps_U': float(valid_df['U_g'].mean()),
        'gbps_noise_evidence': float(valid_df['q_g'].mean()),
        'gbps_active_group_count': int((valid_df['q_g'] > 0).sum()),
        'gbps_valid_group_count': int(len(valid_df)),
        'gbps_active_sample_ratio': active_sample_ratio,
    }


def build_sample_group_score_table(scored_df, group_assignments_df, args, random_state=0):
    if 'sample_idx' not in scored_df.columns:
        raise ValueError('Safe-GBPS requires sample_idx in scored_df')

    merged = scored_df.copy()
    merged['sample_key'] = merged['sample_idx'].astype(int)
    assignments = group_assignments_df.copy()
    assignments['sample_key'] = assignments['sample_key'].astype(int)
    assignments['group_id'] = assignments['group_id'].astype(int)
    merged = merged.merge(assignments[['sample_key', 'group_id']], on='sample_key', how='left', validate='many_to_one')

    if merged['group_id'].isna().any():
        missing_count = int(merged['group_id'].isna().sum())
        raise ValueError(f'missing latent group assignment for {missing_count} samples')
    merged['group_id'] = merged['group_id'].astype(int)

    metrics = []
    sample_frames = []
    for _, group_df in merged.groupby('group_id', sort=True):
        group_metrics, sample_group_df = compute_group_metrics_for_scores(group_df, args=args, random_state=random_state)
        metrics.append(group_metrics)
        sample_frames.append(sample_group_df)

    group_metrics_df = pd.DataFrame(metrics)
    sample_group_df = pd.concat(sample_frames, ignore_index=True) if len(sample_frames) > 0 else pd.DataFrame(columns=merged.columns)
    summary = _aggregate_group_metrics(group_metrics_df, sample_group_df)
    return group_metrics_df, sample_group_df, summary


def _bootstrap_once(sample_group_df, args, random_state):
    sampled_frames = []
    for _, group_df in sample_group_df.groupby('group_id', sort=True):
        group_df = group_df.reset_index(drop=True)
        sampled_index = random_state.randint(0, len(group_df), size=len(group_df))
        sampled_df = group_df.iloc[sampled_index].copy().reset_index(drop=True)
        sampled_frames.append(sampled_df)

    sampled_group_df = pd.concat(sampled_frames, ignore_index=True) if len(sampled_frames) > 0 else pd.DataFrame(columns=sample_group_df.columns)
    metrics = []
    refreshed_sample_frames = []
    for _, group_df in sampled_group_df.groupby('group_id', sort=True):
        group_metrics, refreshed_group_df = compute_group_metrics_for_scores(group_df, args=args, random_state=random_state.randint(0, 10**9))
        metrics.append(group_metrics)
        refreshed_sample_frames.append(refreshed_group_df)
    metrics_df = pd.DataFrame(metrics)
    refreshed_sample_group_df = pd.concat(refreshed_sample_frames, ignore_index=True) if len(refreshed_sample_frames) > 0 else pd.DataFrame(columns=sampled_group_df.columns)
    return _aggregate_group_metrics(metrics_df, refreshed_sample_group_df)


def compute_gbps_u_with_bootstrap(scored_df, group_assignments_df, B=DEFAULT_BOOTSTRAP_B, args=None):
    if args is None:
        raise ValueError('args is required for Safe-GBPS computation')

    group_metrics_df, sample_group_df, summary = build_sample_group_score_table(scored_df, group_assignments_df, args=args, random_state=0)

    bootstrap_rows = []
    bootstrap_values = []
    bootstrap_B = max(1, int(B))
    random_state = np.random.RandomState(0)
    for bootstrap_id in range(bootstrap_B):
        bootstrap_summary = _bootstrap_once(sample_group_df, args=args, random_state=random_state)
        bootstrap_rows.append({
            'bootstrap_id': int(bootstrap_id),
            'U_b': float(bootstrap_summary['gbps_U']),
            'active_group_count_b': int(bootstrap_summary['gbps_active_group_count']),
            'noise_evidence_b': float(bootstrap_summary['gbps_noise_evidence']),
        })
        bootstrap_values.append(float(bootstrap_summary['gbps_U']))

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    gbps_se = float(np.std(np.asarray(bootstrap_values, dtype=float), ddof=0)) if len(bootstrap_values) > 0 else 0.0
    gbps_u_mean = float(np.mean(np.asarray(bootstrap_values, dtype=float))) if len(bootstrap_values) > 0 else float(summary['gbps_U'])

    summary.update({
        'gbps_U_bootstrap_mean': gbps_u_mean,
        'gbps_SE': gbps_se,
    })
    return {
        'gbps_U': float(summary['gbps_U']),
        'gbps_SE': float(summary['gbps_SE']),
        'group_metrics_df': group_metrics_df,
        'sample_group_scores_df': sample_group_df,
        'bootstrap_df': bootstrap_df,
        'bootstrap_values': bootstrap_values,
        'global_noise_evidence': float(summary['gbps_noise_evidence']),
        'summary': summary,
    }
