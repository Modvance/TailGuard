"""Build the compact, public artifact set for a completed TailGuard run."""

import hashlib
import json
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd


FINAL_FILENAMES = (
    'reconciled_memory.pt',
    'coverage_memory.pt',
    'train_audit.csv',
    'training_dynamics.csv',
    'test_scores.csv',
    'per_class_metrics.csv',
    'run_summary.json',
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _read_json(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(_json_safe(payload), file, indent=2, ensure_ascii=False)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError('required TailGuard artifact is missing: {}'.format(path))
    return path


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _merge_new_columns(base, addition, key='sample_idx'):
    if addition is None or len(addition) == 0:
        return base
    if key not in addition.columns:
        raise ValueError('audit table is missing merge key {}.'.format(key))
    if addition[key].duplicated().any():
        raise ValueError('audit table contains duplicate {} values'.format(key))
    new_columns = [column for column in addition.columns if column == key or column not in base.columns]
    if new_columns == [key]:
        return base
    return base.merge(addition[new_columns], on=key, how='left', validate='one_to_one')


def _build_train_audit(work_dir):
    prepare_dir = os.path.join(work_dir, 'prepare')
    stage2_dir = os.path.join(work_dir, 'stage2')
    attachment_dir = os.path.join(work_dir, 'attachment')
    pseudoclass_dir = os.path.join(work_dir, 'pseudoclasses')

    audit = pd.read_csv(_require_file(os.path.join(prepare_dir, 'tailguard_train_metadata.csv')))
    if audit['sample_idx'].duplicated().any():
        raise ValueError('training metadata contains duplicate sample_idx values')

    optional_tables = (
        os.path.join(prepare_dir, 'tailguard_train_analysis_metadata.csv'),
        os.path.join(stage2_dir, 'h_prune_decisions.csv'),
        os.path.join(attachment_dir, 'tail_reconciliation_scores.csv'),
        os.path.join(pseudoclass_dir, 'pseudo_class_members.csv'),
    )
    for path in optional_tables:
        if os.path.isfile(path):
            audit = _merge_new_columns(audit, pd.read_csv(path))

    retained_path = _require_file(os.path.join(stage2_dir, 'stage2_retained_samples.csv'))
    removed_path = _require_file(os.path.join(stage2_dir, 'stage2_removed_samples.csv'))
    retained_ids = set(pd.read_csv(retained_path)['sample_idx'].astype(int).tolist())
    removed_ids = set(pd.read_csv(removed_path)['sample_idx'].astype(int).tolist())
    all_ids = set(audit['sample_idx'].astype(int).tolist())
    if retained_ids & removed_ids:
        raise ValueError('retained and removed training samples overlap')
    if retained_ids | removed_ids != all_ids:
        raise ValueError('retained and removed samples do not cover the training set')
    audit['final_training_status'] = np.where(
        audit['sample_idx'].astype(int).isin(retained_ids),
        'retained',
        'removed',
    )
    return audit.sort_values('sample_idx', kind='mergesort').reset_index(drop=True)


def _build_training_dynamics(work_dir, selected_iter):
    gbps_dir = os.path.join(work_dir, 'gbps')
    frames = []
    if os.path.isdir(gbps_dir):
        for name in sorted(os.listdir(gbps_dir)):
            match = re.fullmatch(r'iter_(\d+)', name)
            if match is None:
                continue
            iteration = int(match.group(1))
            iteration_dir = os.path.join(gbps_dir, name)
            metrics_path = os.path.join(iteration_dir, 'h_group_metrics.csv')
            summary_path = os.path.join(iteration_dir, 'summary.json')
            if not os.path.isfile(metrics_path) or not os.path.isfile(summary_path):
                continue
            frame = pd.read_csv(metrics_path)
            summary = _read_json(summary_path)
            frame.insert(0, 'iteration', iteration)
            for key in (
                'gbps_U',
                'gbps_SE',
                'gbps_noise_evidence',
                'gbps_best_U',
                'gbps_best_iter',
                'gbps_checks_after_best',
                'gbps_improved',
                'gbps_status',
                'gbps_triggered',
            ):
                frame['global_{}'.format(key)] = summary.get(key)
            frame['is_selected_iteration'] = bool(
                selected_iter is not None and iteration == int(selected_iter)
            )
            frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    return pd.DataFrame(columns=['iteration', 'group_id', 'is_selected_iteration'])


def _test_key(frame):
    output = frame.copy()
    normalized = output['img_path'].astype(str).str.replace('\\', '/', regex=False)
    if not normalized.str.contains('/test/', regex=False).all():
        raise ValueError('test score table contains a path without /test/')
    output['test_suffix'] = normalized.str.split('/test/', n=1).str[1]
    output['class_name'] = output['class_name'].astype(str)
    output['label'] = output['label'].astype(int)
    keys = ['class_name', 'test_suffix', 'label']
    if output.duplicated(keys).any():
        raise ValueError('test score table contains duplicate test identities')
    return output, keys


def _prefixed_score_table(path, prefix):
    frame, keys = _test_key(pd.read_csv(_require_file(path)))
    identity_columns = set(keys + ['class_id'])
    rename = {
        column: '{}_{}'.format(prefix, column)
        for column in frame.columns
        if column not in identity_columns
    }
    return frame.rename(columns=rename), keys


def _build_test_scores(work_dir):
    reconciled, keys = _prefixed_score_table(
        os.path.join(work_dir, 'memory', 'memory_eval_scores.csv'),
        'reconciled',
    )
    coverage, _ = _prefixed_score_table(
        os.path.join(work_dir, 'coverage', 'coverage_eval_scores.csv'),
        'coverage',
    )
    coverage = coverage.drop(columns=['class_id'], errors='ignore')
    merged = reconciled.merge(coverage, on=keys, how='outer', validate='one_to_one', indicator=True)
    if not (merged['_merge'] == 'both').all():
        raise ValueError('coverage and reconciled test identities differ')
    merged = merged.drop(columns=['_merge'])
    merged['img_path'] = merged['reconciled_img_path']
    merged = merged.drop(
        columns=['reconciled_img_path', 'coverage_img_path'],
        errors='ignore',
    )
    merged['final_score'] = 0.5 * (
        merged['coverage_final_score'].astype(float)
        + merged['reconciled_final_score'].astype(float)
    )
    return merged.sort_values(keys, kind='mergesort').reset_index(drop=True)


def _build_per_class_metrics(work_dir):
    final_path = _require_file(os.path.join(work_dir, 'final', 'dual_per_class.csv'))
    output = pd.read_csv(final_path).rename(columns={'I-AUROC': 'final_I-AUROC'})

    coverage_path = _require_file(os.path.join(work_dir, 'coverage', 'coverage_per_class_metrics.csv'))
    coverage = pd.read_csv(coverage_path).rename(columns={
        'I-AUROC': 'coverage_I-AUROC',
        'num_samples': 'coverage_num_samples',
    })
    coverage_columns = [column for column in coverage.columns if column != 'class_id']
    output = output.merge(coverage[coverage_columns], on='class_name', how='left', validate='one_to_one')

    memory_summary = _read_json(_require_file(
        os.path.join(work_dir, 'memory', 'memory_eval_summary.json')
    ))
    fused = (memory_summary.get('metrics_by_mode') or {}).get('fused') or {}
    reconciled_rows = fused.get('per_class_metrics') or []
    if reconciled_rows:
        reconciled = pd.DataFrame(reconciled_rows)
        reconciled = reconciled.drop(columns=['class_id'], errors='ignore')
        reconciled = reconciled.rename(columns={
            column: 'reconciled_{}'.format(column)
            for column in reconciled.columns
            if column not in {'class_name'}
        })
        output = output.merge(reconciled, on='class_name', how='left', validate='one_to_one')

    denoising_path = os.path.join(work_dir, 'analysis', 'denoising_diagnostics_by_class.csv')
    if os.path.isfile(denoising_path):
        denoising = pd.read_csv(denoising_path).drop(columns=['class_id'], errors='ignore')
        denoising = denoising.rename(columns={
            column: 'denoising_{}'.format(column)
            for column in denoising.columns
            if column != 'class_name'
        })
        output = output.merge(denoising, on='class_name', how='left', validate='one_to_one')
    return output.sort_values('class_name', kind='mergesort').reset_index(drop=True)


def _compact_summary(complete_summary, output_paths, file_hashes):
    head_prune_summary = dict(complete_summary.get('head_prune_summary') or {})
    head_prune_summary.pop('group_counts', None)
    trigger_summary = dict(complete_summary.get('gbps_trigger_summary') or {})
    trigger_summary.pop('score_source_path', None)
    trigger_summary.pop('selected_checkpoint_path', None)
    final_detection = dict(complete_summary.get('final_detection') or {})
    final_detection.pop('artifacts', None)
    final_detection.pop('pixel_metrics', None)
    coverage_summary = dict(final_detection.get('coverage_summary') or {})
    if coverage_summary:
        coverage_summary.pop('feature_cache_dir', None)
        coverage_summary['feature_cache_disposition'] = (
            'removed with the completed run work directory'
        )
        final_detection['coverage_summary'] = coverage_summary
    return {
        'schema_version': complete_summary.get('schema_version'),
        'method_name': complete_summary.get('method_name'),
        'config_profile': complete_summary.get('config_profile'),
        'variant': complete_summary.get('variant'),
        'resolved_method_config': complete_summary.get('resolved_method_config'),
        'dataset_provenance': complete_summary.get('dataset_provenance'),
        'dhp': {
            'trigger': trigger_summary,
            'purification': head_prune_summary,
        },
        'trp': complete_summary.get('stage2_summary'),
        'pseudo_classes': complete_summary.get('pseudo_class_summary'),
        'denoising_analysis': complete_summary.get('denoising_diagnostics_summary'),
        'memory': {
            'status': complete_summary.get('memory_status'),
            'config': complete_summary.get('memory_config'),
            'num_pseudo_classes': complete_summary.get('num_pseudo_classes'),
            'num_head_pseudo_classes': complete_summary.get('num_head_pseudo_classes'),
            'num_tail_pseudo_classes': complete_summary.get('num_tail_pseudo_classes'),
            'num_tail_memory_banks': complete_summary.get('num_tail_memory_banks'),
        },
        'metrics': {
            'reconciled': complete_summary.get('memory_eval_summary'),
            'final': final_detection,
        },
        'efficiency': {
            'training_time_s': complete_summary.get('total_time_s'),
            'end_to_end_time_s': complete_summary.get('end_to_end_time_s'),
            'time_per_iter_s': complete_summary.get('time_per_iter_s'),
            'iters_per_sec': complete_summary.get('iters_per_sec'),
            'samples_per_sec': complete_summary.get('samples_per_sec'),
        },
        'artifacts': {
            name: {
                'path': output_paths[name],
                'sha256': file_hashes.get(name),
            }
            for name in output_paths
        },
    }


def compact_completed_run(args, complete_summary):
    """Publish seven canonical TailGuard artifacts, then remove ``.work``."""
    tailguard_dir = os.path.realpath(args.tg_root_dir)
    work_dir = os.path.realpath(args.tg_work_dir)
    expected_work_dir = os.path.realpath(os.path.join(tailguard_dir, '.work'))
    if work_dir != expected_work_dir:
        raise ValueError('refusing to compact an unexpected work directory: {}'.format(work_dir))
    if not os.path.isdir(work_dir):
        raise FileNotFoundError('TailGuard work directory is missing: {}'.format(work_dir))

    trigger = complete_summary.get('gbps_trigger_summary') or {}
    train_audit = _build_train_audit(work_dir)
    dynamics = _build_training_dynamics(work_dir, trigger.get('gbps_selected_iter'))
    test_scores = _build_test_scores(work_dir)
    per_class = _build_per_class_metrics(work_dir)

    temporary = tempfile.mkdtemp(prefix='.compact-inprogress-', dir=tailguard_dir)
    try:
        source_reconciled = _require_file(os.path.join(
            work_dir, 'memory', 'pseudo_class_memory_system.pt'
        ))
        source_coverage = _require_file(os.path.join(
            work_dir, 'coverage', 'coverage_memory_system.pt'
        ))
        _link_or_copy(source_reconciled, os.path.join(temporary, 'reconciled_memory.pt'))
        _link_or_copy(source_coverage, os.path.join(temporary, 'coverage_memory.pt'))
        train_audit.to_csv(os.path.join(temporary, 'train_audit.csv'), index=False)
        dynamics.to_csv(os.path.join(temporary, 'training_dynamics.csv'), index=False)
        test_scores.to_csv(os.path.join(temporary, 'test_scores.csv'), index=False)
        per_class.to_csv(os.path.join(temporary, 'per_class_metrics.csv'), index=False)

        if len(train_audit) == 0 or len(test_scores) == 0 or len(per_class) == 0:
            raise ValueError('compact TailGuard outputs contain an empty required table')
        if train_audit['sample_idx'].duplicated().any():
            raise ValueError('compact train audit contains duplicate samples')

        output_paths = {
            name: os.path.join('tailguard', name)
            for name in FINAL_FILENAMES
        }
        output_paths['final_model.pt'] = 'final_model.pt'
        hash_names = FINAL_FILENAMES[:-1]
        file_hashes = {
            name: _sha256(os.path.join(temporary, name))
            for name in hash_names
        }
        final_model_path = _require_file(os.path.join(
            os.path.realpath(os.path.join(args.save_dir, args.save_name)),
            'final_model.pt',
        ))
        file_hashes['final_model.pt'] = _sha256(final_model_path)
        compact_summary = _compact_summary(complete_summary, output_paths, file_hashes)
        _write_json(os.path.join(temporary, 'run_summary.json'), compact_summary)

        for name in FINAL_FILENAMES:
            destination = os.path.join(tailguard_dir, name)
            if os.path.exists(destination):
                raise FileExistsError('compact output already exists: {}'.format(destination))
        for name in FINAL_FILENAMES:
            source = _require_file(os.path.join(temporary, name))
            destination = os.path.join(tailguard_dir, name)
            os.replace(source, destination)
        os.rmdir(temporary)

        published = [os.path.join(tailguard_dir, name) for name in FINAL_FILENAMES]
        if not all(os.path.isfile(path) for path in published):
            raise RuntimeError('compact TailGuard output validation failed after publication')
        shutil.rmtree(work_dir)
        return compact_summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
