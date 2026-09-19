import os

import numpy as np
import pandas as pd


def summarize_contamination_labels(df, prefix=None):
    if df is None or len(df) == 0:
        counts = {
            'num_samples': 0,
            'num_labeled_clean': 0,
            'num_labeled_contaminated': 0,
            'num_unlabeled_samples': 0,
        }
    elif 'is_contaminated' not in df.columns:
        counts = {
            'num_samples': int(len(df)),
            'num_labeled_clean': 0,
            'num_labeled_contaminated': 0,
            'num_unlabeled_samples': int(len(df)),
        }
    else:
        labels = pd.to_numeric(df['is_contaminated'], errors='coerce')
        counts = {
            'num_samples': int(len(df)),
            'num_labeled_clean': int((labels == 0).sum()),
            'num_labeled_contaminated': int((labels == 1).sum()),
            'num_unlabeled_samples': int((~labels.isin([0, 1])).sum()),
        }
    if prefix is None:
        return counts
    return {f'{prefix}_{key}': value for key, value in counts.items()}


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
        return None
    return float(value)


def evaluate_gbps_ci_peak_trigger(U_t, SE_t, noise_evidence, best_U, best_SE, best_iter,
                                  best_noise_evidence, best_check_count, check_count, current_iter,
                                  ci_z, improve_eps, min_checks_before_trigger,
                                  min_checks_after_best, min_noise_evidence, force_trigger=False):
    U_t = float(U_t)
    SE_t = float(SE_t)
    noise_evidence = float(noise_evidence)
    check_count = int(check_count)
    current_iter = int(current_iter)
    ci_z = float(ci_z)
    improve_eps = float(improve_eps)
    min_checks_before_trigger = int(min_checks_before_trigger)
    min_checks_after_best = int(min_checks_after_best)
    min_noise_evidence = float(min_noise_evidence)

    previous_best_U = None if best_U is None else float(best_U)
    previous_best_SE = None if best_SE is None else float(best_SE)
    previous_best_iter = None if best_iter is None else int(best_iter)
    previous_best_noise_evidence = None if best_noise_evidence is None else float(best_noise_evidence)
    previous_best_check_count = None if best_check_count is None else int(best_check_count)

    improved = previous_best_U is None or U_t > (previous_best_U + improve_eps)
    if improved:
        best_U = U_t
        best_SE = SE_t
        best_iter = current_iter
        best_noise_evidence = noise_evidence
        best_check_count = check_count
    else:
        best_U = previous_best_U
        best_SE = previous_best_SE
        best_iter = previous_best_iter
        best_noise_evidence = previous_best_noise_evidence
        best_check_count = previous_best_check_count

    current_upper = float(U_t + ci_z * SE_t)
    best_lower = None if best_U is None or best_SE is None else float(best_U - ci_z * best_SE)
    checks_after_best = 0 if best_check_count is None else max(0, check_count - int(best_check_count))
    enough_checks_before_trigger = check_count >= min_checks_before_trigger
    enough_checks_after_best = checks_after_best >= min_checks_after_best
    ci_peak_ready = (
        previous_best_U is not None
        and best_lower is not None
        and enough_checks_before_trigger
        and enough_checks_after_best
        and current_upper < best_lower
    )
    has_noise_evidence = best_noise_evidence is not None and float(best_noise_evidence) >= min_noise_evidence

    if force_trigger:
        if best_iter is not None and has_noise_evidence:
            selected_iter = int(best_iter)
            status = 'forced_best'
        else:
            selected_iter = None
            status = 'no_noise_forced'
    elif ci_peak_ready and best_iter is not None and has_noise_evidence:
        selected_iter = int(best_iter)
        status = 'ci_peak'
    else:
        selected_iter = None
        status = 'none'

    summary = {
        'stage': 'gbps',
        'iteration': current_iter,
        'gbps_U': U_t,
        'gbps_SE': SE_t,
        'gbps_noise_evidence': noise_evidence,
        'gbps_best_U': _safe_float(best_U),
        'gbps_best_SE': _safe_float(best_SE),
        'gbps_best_iter': None if best_iter is None else int(best_iter),
        'gbps_best_noise_evidence': _safe_float(best_noise_evidence),
        'gbps_check_count': check_count,
        'gbps_best_check_count': None if best_check_count is None else int(best_check_count),
        'gbps_checks_after_best': int(checks_after_best),
        'gbps_current_upper': _safe_float(current_upper),
        'gbps_best_lower': _safe_float(best_lower),
        'gbps_ci_z': ci_z,
        'gbps_improve_eps': improve_eps,
        'gbps_min_checks_before_trigger': min_checks_before_trigger,
        'gbps_min_checks_after_best': min_checks_after_best,
        'gbps_min_noise_evidence': min_noise_evidence,
        'gbps_has_noise_evidence': bool(has_noise_evidence),
        'gbps_improved': bool(improved),
        'gbps_enough_checks_before_trigger': bool(enough_checks_before_trigger),
        'gbps_enough_checks_after_best': bool(enough_checks_after_best),
        'gbps_ci_peak_ready': bool(ci_peak_ready),
        'gbps_status': status,
        'gbps_triggered': bool(status != 'none'),
        'gbps_selected_iter': None if selected_iter is None else int(selected_iter),
        'force_triggered': bool(force_trigger),
    }

    return {
        'status': status,
        'triggered': bool(status != 'none'),
        'selected_iter': selected_iter,
        'best_U': best_U,
        'best_SE': best_SE,
        'best_iter': best_iter,
        'best_noise_evidence': best_noise_evidence,
        'best_check_count': best_check_count,
        'summary': summary,
    }


def load_saved_train_scores(save_dir, iteration):
    score_path = os.path.join(save_dir, 'iter_{:05d}'.format(int(iteration)), 'train_scores.csv')
    if not os.path.isfile(score_path):
        raise FileNotFoundError('saved train scores not found for iter {}: {}'.format(int(iteration), score_path))
    return pd.read_csv(score_path), score_path


def resolve_gbps_best_scored_df(current_scored_df, current_iter, selected_iter, save_dir):
    if selected_iter is None:
        return None, None, None
    if int(selected_iter) == int(current_iter):
        return current_scored_df.copy(), int(current_iter), None
    scored_df, score_path = load_saved_train_scores(save_dir, selected_iter)
    return scored_df.copy(), int(selected_iter), score_path
