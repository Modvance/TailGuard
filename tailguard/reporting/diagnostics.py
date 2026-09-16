"""Analysis-only diagnostics for TailGuard decisions and pseudo-classes."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


ANALYSIS_COLUMNS = ['class_id', 'class_name', 'is_contaminated']


def _safe_ratio(numerator, denominator):
    if denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _empty_counts(num_samples=0):
    return {
        'num_samples': int(num_samples),
        'num_labeled_clean': 0,
        'num_labeled_contaminated': 0,
        'num_unlabeled_samples': int(num_samples),
    }


def _label_counts(df: pd.DataFrame) -> Dict:
    if len(df) == 0 or 'is_contaminated' not in df.columns:
        return _empty_counts(len(df))
    labels = pd.to_numeric(df['is_contaminated'], errors='coerce')
    return {
        'num_samples': int(len(df)),
        'num_labeled_clean': int((labels == 0).sum()),
        'num_labeled_contaminated': int((labels == 1).sum()),
        'num_unlabeled_samples': int((~labels.isin([0, 1])).sum()),
    }


def _ensure_sample_idx(df: Optional[pd.DataFrame], sample_idx_only: bool = False) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=['sample_idx'])
    if 'sample_idx' not in df.columns:
        # Empty partitions can be schema-less after concatenation; normalize them
        # so downstream diagnostics still receive the required key column.
        if len(df) == 0:
            return pd.DataFrame(columns=['sample_idx'])
        raise ValueError('denoising diagnostics require sample_idx')
    output = df[['sample_idx']].copy() if sample_idx_only else df.copy()
    output['sample_idx'] = pd.to_numeric(output['sample_idx'], errors='raise').astype(int)
    return output


def _analysis_view(initial_df: pd.DataFrame) -> pd.DataFrame:
    columns = ['sample_idx'] + [column for column in ANALYSIS_COLUMNS if column in initial_df.columns]
    output = initial_df[columns].copy()
    for column in ANALYSIS_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    return output


def _with_analysis_metadata(samples_df: Optional[pd.DataFrame], analysis_df: pd.DataFrame) -> pd.DataFrame:
    output = _ensure_sample_idx(samples_df, sample_idx_only=True)
    analysis_view = analysis_df.drop_duplicates('sample_idx', keep='first')
    return output.merge(analysis_view, on='sample_idx', how='left', validate='many_to_one')


def _partition_integrity(initial_df: pd.DataFrame,
                         retained_df: pd.DataFrame,
                         removed_df: pd.DataFrame) -> Dict:
    initial_ids = set(initial_df['sample_idx'].tolist())
    retained_ids = set(retained_df['sample_idx'].tolist())
    removed_ids = set(removed_df['sample_idx'].tolist())
    overlap_ids = retained_ids & removed_ids
    covered_ids = retained_ids | removed_ids
    return {
        'initial_sample_idx_unique': bool(initial_df['sample_idx'].is_unique),
        'retained_sample_idx_unique': bool(retained_df['sample_idx'].is_unique),
        'removed_sample_idx_unique': bool(removed_df['sample_idx'].is_unique),
        'retained_removed_disjoint': len(overlap_ids) == 0,
        'partition_covers_initial': covered_ids == initial_ids,
        'num_initial': int(len(initial_ids)),
        'num_covered_initial': int(len(covered_ids & initial_ids)),
        'num_unassigned_initial': int(len(initial_ids - covered_ids)),
        'num_partition_samples_outside_initial': int(len(covered_ids - initial_ids)),
        'num_retained_removed_overlap': int(len(overlap_ids)),
    }


def _contamination_summary(initial_counts: Dict,
                           retained_counts: Dict,
                           removed_counts: Dict,
                           include_cleanup_metrics: bool = True) -> Tuple[Dict, Dict, bool]:
    num_labeled = (
        initial_counts['num_labeled_clean']
        + initial_counts['num_labeled_contaminated']
    )
    labels_available = num_labeled > 0
    if not labels_available:
        return ({
            'noise_before_cleanup': None,
            'noise_removed': None,
            'residual_noise': None,
            'clean_before_cleanup': None,
            'clean_removed': None,
            'clean_retained': None,
        }, {
            'noise_removal_recall': None,
            'residual_noise_rate_of_initial_noise': None,
            'residual_contamination_rate': None,
            'removed_set_noise_precision': None,
            'clean_retention_rate': None,
            'clean_removal_rate': None,
            'removal_rate': None,
        }, False)

    if not include_cleanup_metrics:
        return ({
            'noise_before_cleanup': int(initial_counts['num_labeled_contaminated']),
            'noise_removed': None,
            'residual_noise': None,
            'clean_before_cleanup': int(initial_counts['num_labeled_clean']),
            'clean_removed': None,
            'clean_retained': None,
        }, {
            'noise_removal_recall': None,
            'residual_noise_rate_of_initial_noise': None,
            'residual_contamination_rate': None,
            'removed_set_noise_precision': None,
            'clean_retention_rate': None,
            'clean_removal_rate': None,
            'removal_rate': None,
        }, True)

    noise_before = initial_counts['num_labeled_contaminated']
    noise_removed = removed_counts['num_labeled_contaminated']
    residual_noise = retained_counts['num_labeled_contaminated']
    clean_before = initial_counts['num_labeled_clean']
    clean_removed = removed_counts['num_labeled_clean']
    clean_retained = retained_counts['num_labeled_clean']
    removed_labeled = clean_removed + noise_removed
    retained_labeled = clean_retained + residual_noise
    return ({
        'noise_before_cleanup': int(noise_before),
        'noise_removed': int(noise_removed),
        'residual_noise': int(residual_noise),
        'clean_before_cleanup': int(clean_before),
        'clean_removed': int(clean_removed),
        'clean_retained': int(clean_retained),
    }, {
        'noise_removal_recall': _safe_ratio(noise_removed, noise_before),
        'residual_noise_rate_of_initial_noise': _safe_ratio(residual_noise, noise_before),
        'residual_contamination_rate': _safe_ratio(residual_noise, retained_labeled),
        'removed_set_noise_precision': _safe_ratio(noise_removed, removed_labeled),
        'clean_retention_rate': _safe_ratio(clean_retained, clean_before),
        'clean_removal_rate': _safe_ratio(clean_removed, clean_before),
        'removal_rate': _safe_ratio(removed_counts['num_samples'], initial_counts['num_samples']),
    }, True)


def _source_summary(samples_df: Optional[pd.DataFrame], analysis_df: pd.DataFrame) -> Dict:
    return _label_counts(_with_analysis_metadata(samples_df, analysis_df))


def _removal_source_integrity(removed_df: pd.DataFrame,
                              h_removed_samples_df: Optional[pd.DataFrame],
                              tail_head_noise_samples_df: Optional[pd.DataFrame]) -> Dict:
    removed_ids = set(removed_df['sample_idx'].tolist())
    h_removed_ids = set(_ensure_sample_idx(h_removed_samples_df, sample_idx_only=True)['sample_idx'].tolist())
    tail_noise_ids = set(_ensure_sample_idx(tail_head_noise_samples_df, sample_idx_only=True)['sample_idx'].tolist())
    overlap_ids = h_removed_ids & tail_noise_ids
    return {
        'h_removed_subset_of_total_removed': h_removed_ids <= removed_ids,
        'tail_head_noise_subset_of_total_removed': tail_noise_ids <= removed_ids,
        'removal_sources_disjoint': len(overlap_ids) == 0,
        'num_h_removed_outside_total_removed': int(len(h_removed_ids - removed_ids)),
        'num_tail_head_noise_outside_total_removed': int(len(tail_noise_ids - removed_ids)),
        'num_removal_source_overlap': int(len(overlap_ids)),
    }


def _build_class_rows(initial_df: pd.DataFrame,
                      retained_df: pd.DataFrame,
                      removed_df: pd.DataFrame,
                      include_cleanup_metrics: bool) -> pd.DataFrame:
    if len(initial_df) == 0 or initial_df['class_id'].isna().all():
        return pd.DataFrame(columns=[
            'class_id', 'class_name', 'num_initial', 'num_removed', 'num_retained',
            'noise_before_cleanup', 'noise_removed', 'residual_noise',
            'noise_removal_recall', 'residual_contamination_rate',
            'removed_set_noise_precision', 'clean_retention_rate',
        ])
    rows = []
    group_columns = ['class_id', 'class_name']
    grouped_initial_df = initial_df.copy()
    grouped_initial_df['class_name'] = grouped_initial_df['class_name'].fillna('')
    for values, initial_class_df in grouped_initial_df.groupby(group_columns, dropna=False, sort=True):
        class_id, class_name = values
        sample_ids = set(initial_class_df['sample_idx'].tolist())
        retained_class_df = retained_df.loc[retained_df['sample_idx'].isin(sample_ids)].copy()
        removed_class_df = removed_df.loc[removed_df['sample_idx'].isin(sample_ids)].copy()
        initial_counts = _label_counts(initial_class_df)
        retained_counts = _label_counts(retained_class_df)
        removed_counts = _label_counts(removed_class_df)
        contamination_counts, metrics, _ = _contamination_summary(
            initial_counts,
            retained_counts,
            removed_counts,
            include_cleanup_metrics=include_cleanup_metrics,
        )
        rows.append({
            'class_id': None if pd.isna(class_id) else int(class_id),
            'class_name': '' if pd.isna(class_name) else str(class_name),
            'num_initial': initial_counts['num_samples'],
            'num_removed': removed_counts['num_samples'],
            'num_retained': retained_counts['num_samples'],
            'num_labeled_clean_initial': initial_counts['num_labeled_clean'],
            'num_labeled_contaminated_initial': initial_counts['num_labeled_contaminated'],
            'num_unlabeled_initial': initial_counts['num_unlabeled_samples'],
            **contamination_counts,
            **metrics,
        })
    return pd.DataFrame(rows)


def build_tailguard_denoising_diagnostics(initial_samples_df: pd.DataFrame,
                                             retained_samples_df: Optional[pd.DataFrame],
                                             removed_samples_df: Optional[pd.DataFrame],
                                             h_removed_samples_df: Optional[pd.DataFrame] = None,
                                             tail_head_noise_samples_df: Optional[pd.DataFrame] = None,
                                             manifest_path: Optional[str] = None,
                                             manifest_requested_path: Optional[str] = None,
                                             label_source: Optional[str] = None,
                                             cleanup_status: str = 'completed',
                                             cleanup_reason: Optional[str] = None):
    initial_df = _ensure_sample_idx(initial_samples_df)
    analysis_df = _analysis_view(initial_df)
    initial_df = _with_analysis_metadata(initial_df, analysis_df)
    retained_df = _with_analysis_metadata(retained_samples_df, analysis_df)
    removed_df = _with_analysis_metadata(removed_samples_df, analysis_df)

    initial_counts = _label_counts(initial_df)
    retained_counts = _label_counts(retained_df)
    removed_counts = _label_counts(removed_df)
    integrity = _partition_integrity(initial_df, retained_df, removed_df)
    partition_is_valid = all([
        integrity['initial_sample_idx_unique'],
        integrity['retained_sample_idx_unique'],
        integrity['removed_sample_idx_unique'],
        integrity['retained_removed_disjoint'],
        integrity['partition_covers_initial'],
    ])
    effective_cleanup_status = str(cleanup_status)
    effective_cleanup_reason = cleanup_reason
    if cleanup_status == 'completed' and not partition_is_valid:
        effective_cleanup_status = 'invalid_partition'
        effective_cleanup_reason = cleanup_reason or 'retained_removed_partition_integrity_failed'
    include_cleanup_metrics = effective_cleanup_status == 'completed'
    contamination_counts, contamination_metrics, labels_available = _contamination_summary(
        initial_counts,
        retained_counts,
        removed_counts,
        include_cleanup_metrics=include_cleanup_metrics,
    )
    if not labels_available:
        label_status = 'labels_unavailable'
    elif initial_counts['num_unlabeled_samples'] > 0:
        label_status = 'partial_labels'
    else:
        label_status = 'labels_available'

    summary = {
        'schema_version': 1,
        'analysis_only': True,
        'not_used_by_tailguard_decision_logic': True,
        'cleanup_status': effective_cleanup_status,
        'cleanup_reason': effective_cleanup_reason,
        'label_provenance': {
            'status': label_status,
            'manifest_path': manifest_path,
            'manifest_requested_path': manifest_requested_path,
            'analysis_metadata_source': label_source,
            'label_column': 'is_contaminated',
            **initial_counts,
        },
        'decision_counts': {
            'num_initial': initial_counts['num_samples'],
            'num_removed': removed_counts['num_samples'],
            'num_retained': retained_counts['num_samples'],
        },
        'contamination_counts': contamination_counts,
        'contamination_metrics': contamination_metrics,
        'removal_breakdown': {
            'h_removed': _source_summary(h_removed_samples_df, analysis_df),
            'tail_head_noise_removed': _source_summary(tail_head_noise_samples_df, analysis_df),
            'total_removed': removed_counts,
        },
        'integrity': integrity,
        'removal_source_integrity': _removal_source_integrity(
            removed_df,
            h_removed_samples_df,
            tail_head_noise_samples_df,
        ),
    }
    return summary, _build_class_rows(
        initial_df,
        retained_df,
        removed_df,
        include_cleanup_metrics=include_cleanup_metrics,
    )


def _safe_mean(values):
    values = pd.Series(values).dropna()
    return None if len(values) == 0 else float(values.mean())


def _safe_median(values):
    values = pd.Series(values).dropna()
    return None if len(values) == 0 else float(values.median())


def _report_embedding_lookup(cls_embeddings_payload: Dict):
    if 'sample_idx' not in cls_embeddings_payload or 'embeddings' not in cls_embeddings_payload:
        raise ValueError('pseudo-class report requires sample_idx and embeddings in cls payload')
    sample_ids = [int(value) for value in cls_embeddings_payload['sample_idx']]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError('pseudo-class report CLS payload contains duplicate sample_idx values')
    embeddings = np.asarray(cls_embeddings_payload['embeddings'].float().cpu(), dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sample_ids):
        raise ValueError('pseudo-class report CLS payload has an invalid embedding shape')
    if not np.isfinite(embeddings).all():
        raise ValueError('pseudo-class report CLS payload contains non-finite embeddings')
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError('pseudo-class report CLS payload contains zero-norm embeddings')
    embeddings = embeddings / norms[:, None]
    return {sample_idx: embeddings[index] for index, sample_idx in enumerate(sample_ids)}


def _empty_pseudoclass_contingency() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'pseudo_class_id', 'pseudo_class_type', 'semantic_label', 'num_samples',
        'pseudo_class_fraction', 'semantic_label_fraction',
    ])


def build_tailguard_pseudoclass_report(members_df: pd.DataFrame,
                                          classes_df: pd.DataFrame,
                                          cls_embeddings_payload: Dict,
                                          analysis_metadata_df: Optional[pd.DataFrame] = None):
    required_member_columns = {
        'sample_idx', 'pseudo_class_id', 'pseudo_class_type', 'source_status',
    }
    required_class_columns = {
        'pseudo_class_id', 'pseudo_class_type', 'num_members',
    }
    missing_members = required_member_columns.difference(members_df.columns)
    missing_classes = required_class_columns.difference(classes_df.columns)
    if missing_members:
        raise ValueError('pseudo-class report members are missing: {}'.format(', '.join(sorted(missing_members))))
    if missing_classes:
        raise ValueError('pseudo-class report classes are missing: {}'.format(', '.join(sorted(missing_classes))))

    members = members_df.copy().reset_index(drop=True)
    classes = classes_df.copy().reset_index(drop=True)
    members['sample_idx'] = members['sample_idx'].astype(int)
    members['pseudo_class_id'] = members['pseudo_class_id'].astype(int)
    classes['pseudo_class_id'] = classes['pseudo_class_id'].astype(int)
    if members['sample_idx'].duplicated().any():
        raise ValueError('pseudo-class report members contain duplicate sample_idx values')
    if classes['pseudo_class_id'].duplicated().any():
        raise ValueError('pseudo-class report classes contain duplicate pseudo_class_id values')
    if (classes['num_members'].astype(int) <= 0).any():
        raise ValueError('pseudo-class report classes contain empty pseudo-classes')
    if not set(members['pseudo_class_id']).issubset(set(classes['pseudo_class_id'])):
        raise ValueError('pseudo-class report members reference unknown pseudo-classes')
    membership_counts = members.groupby('pseudo_class_id').size()
    expected_counts = classes.set_index('pseudo_class_id')['num_members'].astype(int)
    if membership_counts.to_dict() != expected_counts.to_dict():
        raise ValueError('pseudo-class member counts do not match pseudo-class registry')

    if len(members) == 0:
        empty_predictions = members.copy()
        for column in [
            'predicted_pseudo_class_id', 'predicted_pseudo_class_type', 'top1_similarity',
            'second_similarity', 'margin', 'true_prototype_similarity', 'in_sample_agrees',
            'loo_predicted_pseudo_class_id', 'loo_predicted_pseudo_class_type',
            'loo_top1_similarity', 'loo_margin', 'loo_agrees', 'loo_status',
        ]:
            empty_predictions[column] = pd.Series(dtype='object')
        summary = {
            'schema_version': 1,
            'analysis_only': True,
            'not_used_by_tailguard_decision_logic': True,
            'status': 'empty_registry',
            'num_registry_members': 0,
            'num_pseudo_classes': 0,
        }
        return summary, empty_predictions, classes.copy(), _empty_pseudoclass_contingency()

    embedding_lookup = _report_embedding_lookup(cls_embeddings_payload)
    missing_embeddings = sorted(set(members['sample_idx']).difference(embedding_lookup))
    if missing_embeddings:
        raise ValueError('pseudo-class report CLS payload is missing retained samples: {}'.format(missing_embeddings[:10]))
    embeddings = np.stack([embedding_lookup[sample_idx] for sample_idx in members['sample_idx']], axis=0)
    class_ids = classes['pseudo_class_id'].astype(int).tolist()
    class_type_lookup = classes.set_index('pseudo_class_id')['pseudo_class_type'].to_dict()
    member_class_ids = members['pseudo_class_id'].to_numpy(dtype=int)
    class_positions = {class_id: index for index, class_id in enumerate(class_ids)}
    class_sums = np.zeros((len(class_ids), embeddings.shape[1]), dtype=np.float64)
    class_counts = np.zeros(len(class_ids), dtype=int)
    for index, class_id in enumerate(member_class_ids):
        class_position = class_positions[int(class_id)]
        class_sums[class_position] += embeddings[index]
        class_counts[class_position] += 1
    prototype_norms = np.linalg.norm(class_sums, axis=1)
    valid_prototypes = prototype_norms > 1e-12
    prototypes = np.full_like(class_sums, np.nan)
    prototypes[valid_prototypes] = class_sums[valid_prototypes] / prototype_norms[valid_prototypes, None]
    if not valid_prototypes.any():
        raise ValueError('pseudo-class report has no valid CLS prototypes')

    similarities = embeddings @ prototypes[valid_prototypes].T
    valid_class_ids = np.asarray(class_ids, dtype=int)[valid_prototypes]
    best_positions = np.argmax(similarities, axis=1)
    best_class_ids = valid_class_ids[best_positions]
    best_similarities = similarities[np.arange(len(members)), best_positions]
    if similarities.shape[1] > 1:
        second_similarities = np.partition(similarities, -2, axis=1)[:, -2]
    else:
        second_similarities = np.full(len(members), np.nan)
    true_similarities = np.full(len(members), np.nan)
    for index, class_id in enumerate(member_class_ids):
        class_position = class_positions[int(class_id)]
        if valid_prototypes[class_position]:
            true_similarities[index] = float(embeddings[index] @ prototypes[class_position])

    predictions = members.copy()
    predictions['predicted_pseudo_class_id'] = best_class_ids.astype(int)
    predictions['predicted_pseudo_class_type'] = [class_type_lookup[int(class_id)] for class_id in best_class_ids]
    predictions['top1_similarity'] = best_similarities
    predictions['second_similarity'] = second_similarities
    predictions['margin'] = best_similarities - second_similarities
    predictions['true_prototype_similarity'] = true_similarities
    predictions['prototype_status'] = [
        'valid' if valid_prototypes[class_positions[int(class_id)]] else 'zero_norm'
        for class_id in member_class_ids
    ]
    predictions['in_sample_agrees'] = np.where(
        np.isfinite(true_similarities),
        best_class_ids == member_class_ids,
        np.nan,
    )

    loo_predicted_ids = []
    loo_top1_similarities = []
    loo_margins = []
    loo_agrees = []
    loo_statuses = []
    for index, class_id in enumerate(member_class_ids):
        class_position = class_positions[int(class_id)]
        if class_counts[class_position] <= 1:
            loo_predicted_ids.append(np.nan)
            loo_top1_similarities.append(np.nan)
            loo_margins.append(np.nan)
            loo_agrees.append(np.nan)
            loo_statuses.append('singleton_class')
            continue
        loo_prototypes = prototypes.copy()
        own_sum = class_sums[class_position] - embeddings[index]
        own_norm = np.linalg.norm(own_sum)
        if own_norm <= 1e-12:
            loo_predicted_ids.append(np.nan)
            loo_top1_similarities.append(np.nan)
            loo_margins.append(np.nan)
            loo_agrees.append(np.nan)
            loo_statuses.append('zero_norm_loo_prototype')
            continue
        loo_prototypes[class_position] = own_sum / own_norm
        eligible = np.isfinite(loo_prototypes).all(axis=1)
        loo_similarities = embeddings[index] @ loo_prototypes[eligible].T
        loo_class_ids = np.asarray(class_ids, dtype=int)[eligible]
        best_position = int(np.argmax(loo_similarities))
        predicted_class_id = int(loo_class_ids[best_position])
        loo_predicted_ids.append(predicted_class_id)
        loo_top1_similarities.append(float(loo_similarities[best_position]))
        loo_margins.append(
            np.nan if len(loo_similarities) == 1 else float(
                loo_similarities[best_position] - np.partition(loo_similarities, -2)[-2]
            )
        )
        loo_agrees.append(predicted_class_id == int(class_id))
        loo_statuses.append('evaluated')
    predictions['loo_predicted_pseudo_class_id'] = loo_predicted_ids
    predictions['loo_predicted_pseudo_class_type'] = [
        np.nan if pd.isna(class_id) else class_type_lookup[int(class_id)]
        for class_id in loo_predicted_ids
    ]
    predictions['loo_top1_similarity'] = loo_top1_similarities
    predictions['loo_margin'] = loo_margins
    predictions['loo_agrees'] = loo_agrees
    predictions['loo_status'] = loo_statuses

    analysis = pd.DataFrame(columns=['sample_idx'] + ANALYSIS_COLUMNS)
    if analysis_metadata_df is not None:
        analysis_input = _ensure_sample_idx(analysis_metadata_df)
        if analysis_input['sample_idx'].duplicated().any():
            raise ValueError('pseudo-class analysis metadata contains duplicate sample_idx values')
        analysis = _analysis_view(analysis_input)
    predictions = predictions.merge(analysis, on='sample_idx', how='left', validate='one_to_one')

    semantic_label_source = 'unavailable'
    semantic_labels = pd.Series(np.nan, index=predictions.index, dtype='object')
    if 'class_id' in predictions and predictions['class_id'].notna().any():
        semantic_label_source = 'class_id'
        semantic_labels = predictions['class_id'].map(lambda value: None if pd.isna(value) else str(int(value)))
    elif 'class_name' in predictions and predictions['class_name'].notna().any():
        semantic_label_source = 'class_name'
        semantic_labels = predictions['class_name'].map(lambda value: None if pd.isna(value) else str(value))
    predictions['semantic_label'] = semantic_labels

    class_rows = classes.copy()
    in_sample = predictions.loc[predictions['in_sample_agrees'].notna()]
    loo = predictions.loc[predictions['loo_status'] == 'evaluated']
    aggregate_rows = []
    for class_id, class_df in predictions.groupby('pseudo_class_id', sort=True):
        in_sample_class = class_df.loc[class_df['in_sample_agrees'].notna()]
        loo_class = class_df.loc[class_df['loo_status'] == 'evaluated']
        contamination = pd.to_numeric(class_df['is_contaminated'], errors='coerce')
        contamination_known = contamination.isin([0, 1])
        semantic_class = class_df.loc[class_df['semantic_label'].notna(), 'semantic_label']
        majority_label = None
        majority_count = 0
        if len(semantic_class) > 0:
            label_counts = semantic_class.value_counts().sort_index()
            majority_count = int(label_counts.max())
            majority_label = str(label_counts.loc[label_counts == majority_count].index[0])
        aggregate_rows.append({
            'pseudo_class_id': int(class_id),
            'num_in_sample_eligible': int(len(in_sample_class)),
            'in_sample_agreement': _safe_mean(in_sample_class['in_sample_agrees'].astype(float)),
            'mean_true_prototype_similarity': _safe_mean(in_sample_class['true_prototype_similarity']),
            'median_true_prototype_similarity': _safe_median(in_sample_class['true_prototype_similarity']),
            'mean_in_sample_margin': _safe_mean(in_sample_class['margin']),
            'num_loo_evaluated': int(len(loo_class)),
            'loo_agreement': _safe_mean(loo_class['loo_agrees'].astype(float)),
            'mean_loo_margin': _safe_mean(loo_class['loo_margin']),
            'num_semantic_labeled': int(len(semantic_class)),
            'semantic_majority_label': majority_label,
            'semantic_majority_count': int(majority_count),
            'num_contamination_labeled': int(contamination_known.sum()),
            'num_clean': int((contamination == 0).sum()),
            'num_contaminated': int((contamination == 1).sum()),
            'num_contamination_unknown': int((~contamination_known).sum()),
            'contamination_rate': _safe_ratio(int((contamination == 1).sum()), int(contamination_known.sum())),
            'clean_rate': _safe_ratio(int((contamination == 0).sum()), int(contamination_known.sum())),
        })
    class_rows = class_rows.merge(pd.DataFrame(aggregate_rows), on='pseudo_class_id', how='left', validate='one_to_one')

    semantic_df = predictions.loc[predictions['semantic_label'].notna()].copy()
    contingency = _empty_pseudoclass_contingency()
    label_metrics = {
        'status': 'labels_unavailable',
        'semantic_label_source': semantic_label_source,
        'num_labeled_members': 0,
        'num_unlabeled_members': int(len(predictions)),
        'num_semantic_labels': 0,
        'num_pseudo_classes_with_labels': 0,
        'purity': None,
        'majority_map_accuracy': None,
        'adjusted_rand_index': None,
        'normalized_mutual_information': None,
        'homogeneity': None,
        'completeness': None,
        'v_measure': None,
    }
    if len(semantic_df) > 0:
        grouped = semantic_df.groupby(['pseudo_class_id', 'semantic_label'], sort=True).size().rename('num_samples').reset_index()
        grouped = grouped.merge(
            classes[['pseudo_class_id', 'pseudo_class_type']],
            on='pseudo_class_id',
            how='left',
            validate='many_to_one',
        )
        grouped['pseudo_class_fraction'] = grouped['num_samples'] / grouped.groupby('pseudo_class_id')['num_samples'].transform('sum')
        grouped['semantic_label_fraction'] = grouped['num_samples'] / grouped.groupby('semantic_label')['num_samples'].transform('sum')
        contingency = grouped[[
            'pseudo_class_id', 'pseudo_class_type', 'semantic_label', 'num_samples',
            'pseudo_class_fraction', 'semantic_label_fraction',
        ]]
        purity = float(grouped.groupby('pseudo_class_id')['num_samples'].max().sum() / len(semantic_df))
        num_labels = int(semantic_df['semantic_label'].nunique())
        num_pseudo_classes = int(semantic_df['pseudo_class_id'].nunique())
        label_metrics.update({
            'status': 'available' if num_labels > 1 and num_pseudo_classes > 1 else 'degenerate_single_label_or_cluster',
            'num_labeled_members': int(len(semantic_df)),
            'num_unlabeled_members': int(len(predictions) - len(semantic_df)),
            'num_semantic_labels': num_labels,
            'num_pseudo_classes_with_labels': num_pseudo_classes,
            'purity': purity,
            'majority_map_accuracy': purity,
        })
        semantic_values = semantic_df['semantic_label'].tolist()
        pseudo_values = semantic_df['pseudo_class_id'].astype(int).tolist()
        try:
            from sklearn.metrics import (
                adjusted_rand_score,
                homogeneity_completeness_v_measure,
                normalized_mutual_info_score,
            )
            label_metrics['adjusted_rand_index'] = float(adjusted_rand_score(semantic_values, pseudo_values))
            label_metrics['normalized_mutual_information'] = float(
                normalized_mutual_info_score(semantic_values, pseudo_values, average_method='arithmetic')
            )
            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(semantic_values, pseudo_values)
            label_metrics.update({
                'homogeneity': float(homogeneity),
                'completeness': float(completeness),
                'v_measure': float(v_measure),
                'sklearn_metric_status': 'available',
            })
        except (ImportError, AttributeError) as error:
            label_metrics['sklearn_metric_status'] = 'unavailable: {}'.format(error)

    loo_eligible = predictions['loo_status'].isin(['evaluated', 'zero_norm_loo_prototype'])
    summary = {
        'schema_version': 1,
        'analysis_only': True,
        'not_used_by_tailguard_decision_logic': True,
        'status': 'completed',
        'cls_embedding_source': cls_embeddings_payload.get('embedding_source'),
        'registry': {
            'num_registry_members': int(len(members)),
            'num_pseudo_classes': int(len(classes)),
            'num_head_pseudo_classes': int((classes['pseudo_class_type'] == 'head').sum()),
            'num_tail_pseudo_classes': int((classes['pseudo_class_type'] == 'tail').sum()),
            'source_status_counts': {
                str(key): int(value) for key, value in members['source_status'].value_counts().sort_index().items()
            },
            'member_ids_unique': True,
            'class_sizes_match_members': True,
        },
        'prototype_agreement': {
            'num_in_sample_eligible': int(len(in_sample)),
            'in_sample_agreement': _safe_mean(in_sample['in_sample_agrees'].astype(float)),
            'mean_true_prototype_similarity': _safe_mean(in_sample['true_prototype_similarity']),
            'median_true_prototype_similarity': _safe_median(in_sample['true_prototype_similarity']),
            'mean_in_sample_margin': _safe_mean(in_sample['margin']),
            'num_loo_eligible': int(loo_eligible.sum()),
            'num_loo_evaluated': int(len(loo)),
            'num_loo_excluded_zero_norm': int((predictions['loo_status'] == 'zero_norm_loo_prototype').sum()),
            'loo_agreement': _safe_mean(loo['loo_agrees'].astype(float)),
            'mean_loo_margin': _safe_mean(loo['loo_margin']),
        },
        'semantic_label_metrics': label_metrics,
        'contamination_summary': {
            'num_labeled_members': int(pd.to_numeric(predictions['is_contaminated'], errors='coerce').isin([0, 1]).sum()),
            'num_clean': int((pd.to_numeric(predictions['is_contaminated'], errors='coerce') == 0).sum()),
            'num_contaminated': int((pd.to_numeric(predictions['is_contaminated'], errors='coerce') == 1).sum()),
        },
    }
    return summary, predictions, class_rows, contingency
