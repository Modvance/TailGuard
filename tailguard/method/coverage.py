"""Build TailGuard's coverage reference from the trained full pipeline.

This module rebuilds the coverage pseudo-class registry from a full run,
extracts frozen DINOv2 CLS/patch features once, and combines the resulting
coverage memory increment with the reconstruction scores already saved by the
full run. It does not update any model parameters.
"""

import datetime as dt
import hashlib
import json
import math
import os
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# SciPy 1.4 still imports NumPy's removed scalar aliases in the original
# Dinomaly environment.  Restore only the aliases required during legacy
# module import; numerical execution below uses explicit dtypes.
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

from tailguard.data.datasets import get_data_transforms
from tailguard.engine.trainer import (
    _checkpoint_args_get,
    load_model_from_train_checkpoint,
    load_train_checkpoint_metadata,
)
from tailguard.method.cte_replay import (
    encoder_features_once,
    build_memory_system_from_cache,
    score_cached_features,
)
from tailguard.data.profiles import (
    MVTec_PROFILE,
    get_dataset_profile,
    validate_profile_contract,
)
from tailguard.method.cte import route_evidence_mask
from tailguard.method.registry import build_tailguard_pseudoclass_registry


SCHEMA_VERSION = 1
SCORE_KEY_COLUMNS = ('class_name', 'test_suffix', 'label')


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(_json_safe(payload), file, indent=2, ensure_ascii=False)


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _normalized_suffix(path, marker):
    normalized = str(path).replace('\\', '/')
    token = '/{}/'.format(marker.strip('/'))
    if token not in normalized:
        raise ValueError('path does not contain {}: {}'.format(token, path))
    return normalized.split(token, 1)[1]


def _test_key_frame(frame):
    output = frame.copy()
    output['test_suffix'] = output['img_path'].map(lambda value: _normalized_suffix(value, 'test'))
    output['label'] = output['label'].astype(int)
    return output


def _load_full_artifacts(full_run_dir):
    run_dir = os.path.realpath(full_run_dir)
    tailguard_dir = os.path.join(run_dir, 'tailguard')
    paths = {
        'retained': os.path.join(tailguard_dir, 'stage2', 'stage2_retained_samples.csv'),
        'removed': os.path.join(tailguard_dir, 'stage2', 'stage2_removed_samples.csv'),
        'full_members': os.path.join(tailguard_dir, 'pseudoclasses', 'pseudo_class_members.csv'),
        'full_scores': os.path.join(tailguard_dir, 'memory', 'memory_eval_scores.csv'),
    }
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError('{} artifact is missing: {}'.format(name, path))
    retained = pd.read_csv(paths['retained'])
    removed = pd.read_csv(paths['removed'])
    required = {
        'sample_idx', 'sample_key', 'img_path', 'class_id', 'base_idx',
        'tail_candidate', 'adaptive_angle', 'group_id',
    }
    missing = required.difference(retained.columns)
    if missing:
        raise ValueError('retained samples are missing columns: {}'.format(sorted(missing)))
    return run_dir, paths, retained, removed


def _resolve_profile(parsed_args, checkpoint_metadata):
    checkpoint_profile = _checkpoint_args_get(checkpoint_metadata.get('args'), 'dataset_profile', None)
    profile = get_dataset_profile(parsed_args.dataset_profile) if parsed_args.dataset_profile else (
        get_dataset_profile(checkpoint_profile) if checkpoint_profile else MVTec_PROFILE
    )
    validate_profile_contract(
        profile,
        checkpoint_profile,
        _checkpoint_args_get(checkpoint_metadata.get('args'), 'dataset_item_list', None),
        'checkpoint',
    )
    return profile


def _stable_identity_audit(retained, full_members):
    identity_columns = ['sample_idx', 'class_id', 'base_idx']
    left = retained[identity_columns].astype(int).sort_values(identity_columns).reset_index(drop=True)
    right = full_members[identity_columns].astype(int).sort_values(identity_columns).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError('Full pseudo-class registry does not exactly cover Stage 2 retained samples')


def _build_raw_registry(retained, removed, cls_payload):
    h_clean = retained.loc[retained['tail_candidate'].astype(int) == 0].copy()
    tail_open = retained.loc[retained['tail_candidate'].astype(int) == 1].copy()
    tail_affiliated = tail_open.iloc[0:0].copy()
    tail_affiliated['best_group_id'] = pd.Series(dtype=int)
    return build_tailguard_pseudoclass_registry(
        h_clean,
        tail_affiliated,
        tail_open,
        retained,
        removed,
        cls_payload,
    )


def _content_key(path):
    stat = os.stat(path)
    return '{}-{}'.format(int(stat.st_size), _sha256_file(path))


def _cache_record_path(cache_dir, content_key):
    return os.path.join(cache_dir, content_key[:2], '{}.pt'.format(content_key))


def _load_image_batch(paths, transform):
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image.convert('RGB')))
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def ensure_feature_cache(model, paths, transform, cache_dir, device, batch_size):
    os.makedirs(cache_dir, exist_ok=True)
    unique_paths = list(dict.fromkeys(os.path.realpath(path) for path in paths))
    keyed = [(path, _content_key(path)) for path in unique_paths]
    missing = [(path, key) for path, key in keyed if not os.path.isfile(_cache_record_path(cache_dir, key))]
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            images = _load_image_batch([row[0] for row in batch], transform).to(device)
            patches, spatial_size, cls_embeddings, _ = encoder_features_once(model, images)
            for index, (_, key) in enumerate(batch):
                record_path = _cache_record_path(cache_dir, key)
                os.makedirs(os.path.dirname(record_path), exist_ok=True)
                temporary = '{}.tmp-{}'.format(record_path, os.getpid())
                torch.save({
                    'schema_version': SCHEMA_VERSION,
                    'content_key': key,
                    'cls': cls_embeddings[index].detach().cpu().contiguous(),
                    'patches': patches[index].detach().cpu().contiguous(),
                    'spatial_size': tuple(int(value) for value in spatial_size),
                }, temporary)
                try:
                    os.replace(temporary, record_path)
                except FileExistsError:
                    os.unlink(temporary)
            completed = min(start + len(batch), len(missing))
            if completed == len(missing) or completed % max(256, batch_size) < batch_size:
                print('coverage feature cache [{}/{}]'.format(completed, len(missing)), flush=True)
    finally:
        if was_training:
            model.train()
    return {
        'num_requested_paths': len(paths),
        'num_unique_paths': len(unique_paths),
        'num_cache_hits': len(keyed) - len(missing),
        'num_cache_misses': len(missing),
        'path_to_key': {path: key for path, key in keyed},
    }


def _load_cached_record(cache_dir, content_key):
    record = torch.load(_cache_record_path(cache_dir, content_key), map_location='cpu')
    if record.get('content_key') != content_key:
        raise ValueError('feature cache content-key mismatch')
    return record


def _path_map_for_retained(retained, data_path, item_list):
    path_map = {}
    for row in retained.itertuples(index=False):
        class_id = int(row.class_id)
        base_idx = int(row.base_idx)
        class_name = item_list[class_id]
        original_name = os.path.basename(str(row.img_path).replace('\\', '/'))
        path = os.path.realpath(os.path.join(data_path, class_name, 'train', 'good', original_name))
        if not os.path.isfile(path):
            test_dir = os.path.join(data_path, class_name, 'test')
            if os.path.isdir(test_dir):
                for defect_type in sorted(os.listdir(test_dir), key=len, reverse=True):
                    prefix = defect_type + '_'
                    if not original_name.startswith(prefix):
                        continue
                    injected_source = os.path.realpath(os.path.join(
                        test_dir, defect_type, original_name[len(prefix):]
                    ))
                    if os.path.isfile(injected_source):
                        path = injected_source
                        break
        if not os.path.isfile(path):
            raise FileNotFoundError('retained training image is missing: {}'.format(path))
        path_map[(class_id, base_idx)] = path
    if len(path_map) != len(retained):
        raise ValueError('retained identities are not unique')
    return path_map


def _path_map_for_test(full_scores, data_path):
    output = {}
    keyed = _test_key_frame(full_scores)
    for row in keyed.itertuples(index=False):
        path = os.path.realpath(os.path.join(data_path, row.class_name, 'test', row.test_suffix))
        if not os.path.isfile(path):
            raise FileNotFoundError('test image is missing: {}'.format(path))
        key = (str(row.class_name), str(row.test_suffix), int(row.label))
        if key in output:
            raise ValueError('Full scores contain duplicate test identity: {}'.format(key))
        output[key] = path
    return output


def _payload_from_cache(retained, train_paths, path_to_key, cache_dir):
    ordered = retained.sort_values('sample_idx', kind='mergesort')
    embeddings = []
    image_paths = []
    for row in ordered.itertuples(index=False):
        path = train_paths[(int(row.class_id), int(row.base_idx))]
        embeddings.append(_load_cached_record(cache_dir, path_to_key[path])['cls'])
        image_paths.append(str(row.img_path))
    return {
        'sample_idx': ordered['sample_idx'].astype(int).tolist(),
        'img_path': image_paths,
        'embeddings': torch.stack(embeddings, dim=0),
        'embedding_source': 'encoder_cls',
    }


def _memory_cache_for_registry(registry, retained, train_paths, path_to_key, cache_dir):
    required = {
        (int(row.class_id), int(row.base_idx))
        for row in registry['members_df'].itertuples(index=False)
    }
    tail_required = {
        (int(row.class_id), int(row.base_idx))
        for row in registry['members_df'].loc[
            registry['members_df']['pseudo_class_type'] == 'tail'
        ].itertuples(index=False)
    }
    if required != set(train_paths):
        raise ValueError('rebuilt coverage registry does not exactly cover retained identities')
    cache = {}
    for key in sorted(required):
        record = _load_cached_record(cache_dir, path_to_key[train_paths[key]])
        cache[key] = {'cls': record['cls']}
        if key in tail_required:
            cache[key].update({
                'patches': record['patches'],
                'spatial_size': tuple(record['spatial_size']),
            })
    return cache


def _registry_wrapper(registry):
    return {
        'variant': 'coverage',
        'members': registry['members_df'],
        'classes': registry['classes_df'],
    }


def _routing_cache(memory_system, device):
    entries = memory_system['class_entries']
    class_ids = sorted(int(value) for value in entries)
    prototypes = torch.stack([entries[value]['prototype_cls'] for value in class_ids]).to(device)
    return {
        'class_ids': class_ids,
        'prototypes': F.normalize(prototypes.float(), dim=-1),
        'class_types': [str(entries[value]['pseudo_class_type']) for value in class_ids],
    }


@torch.no_grad()
def score_test_from_cache(full_scores, test_paths, path_to_key, cache_dir, memory_system, args, device):
    routing = _routing_cache(memory_system, device)
    memory_device_cache = {}
    rows = []
    keyed_scores = _test_key_frame(full_scores)
    for start in range(0, len(keyed_scores), args.batch_size):
        batch = keyed_scores.iloc[start:start + args.batch_size]
        records = [
            _load_cached_record(cache_dir, path_to_key[test_paths[
                (str(row.class_name), str(row.test_suffix), int(row.label))
            ]])
            for row in batch.itertuples(index=False)
        ]
        patch_features = torch.stack([record['patches'] for record in records]).to(device)
        cls_embeddings = torch.stack([record['cls'] for record in records]).to(device)
        spatial_sizes = {tuple(record['spatial_size']) for record in records}
        if len(spatial_sizes) != 1:
            raise ValueError('test feature cache contains mixed spatial sizes')
        memory = score_cached_features(
            patch_features,
            next(iter(spatial_sizes)),
            cls_embeddings,
            memory_system,
            args,
            device,
            routing,
            memory_device_cache,
        )
        applied = route_evidence_mask(
            memory['route_eligible'],
            memory['similarity_margin'],
            args.tg_memory_route_margin_threshold,
        )
        for index, row in enumerate(batch.itertuples(index=False)):
            reconstruction = float(row.recon_score)
            memory_score = float(memory['memory_scores'][index].item())
            final_score = reconstruction + args.tg_memory_fusion_lambda * memory_score if bool(applied[index]) else reconstruction
            rows.append({
                'class_id': int(row.class_id),
                'class_name': str(row.class_name),
                'img_path': str(row.img_path),
                'label': int(row.label),
                'recon_score': reconstruction,
                'memory_score': memory_score,
                'final_score': final_score,
                'route_eligible': int(memory['route_eligible'][index].item()),
                'mem_applied': int(applied[index].item()),
                'predicted_pseudo_class_id': int(memory['predicted_pseudo_class_id'][index].item()),
                'predicted_pseudo_class_type': str(memory['predicted_pseudo_class_type'][index]),
                'top1_similarity': float(memory['top1_similarity'][index].item()),
                'second_similarity': float(memory['second_similarity'][index].item()),
                'similarity_margin': float(memory['similarity_margin'][index].item()),
            })
        completed = min(start + len(batch), len(keyed_scores))
        if completed == len(keyed_scores) or completed % max(256, args.batch_size) < args.batch_size:
            print('coverage test scores [{}/{}]'.format(completed, len(keyed_scores)), flush=True)
    return pd.DataFrame(rows)


def _macro_image_auroc(score_df):
    from sklearn.metrics import roc_auc_score

    rows = []
    for (class_id, class_name), frame in score_df.groupby(['class_id', 'class_name'], sort=True):
        rows.append({
            'class_id': int(class_id),
            'class_name': str(class_name),
            'I-AUROC': float(roc_auc_score(frame['label'].astype(int), frame['final_score'].astype(float))),
            'num_samples': int(len(frame)),
        })
    return pd.DataFrame(rows), float(np.mean([row['I-AUROC'] for row in rows]))


def _frame_comparison(actual, expected, key_columns, float_atol, name):
    actual = actual.copy()
    expected = expected.copy()
    merged = actual.merge(expected, on=key_columns, how='outer', suffixes=('_actual', '_expected'), indicator=True)
    both = merged['_merge'] == 'both'
    report = {
        'name': name,
        'actual_rows': len(actual),
        'expected_rows': len(expected),
        'matched_rows': int(both.sum()),
        'missing_rows': int((merged['_merge'] == 'right_only').sum()),
        'unexpected_rows': int((merged['_merge'] == 'left_only').sum()),
        'columns': {},
    }
    shared_columns = sorted(set(actual.columns).intersection(expected.columns).difference(key_columns))
    exact = report['missing_rows'] == 0 and report['unexpected_rows'] == 0
    for column in shared_columns:
        left = merged.loc[both, '{}_actual'.format(column)]
        right = merged.loc[both, '{}_expected'.format(column)]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = pd.to_numeric(left, errors='coerce').to_numpy(dtype=float)
            right_values = pd.to_numeric(right, errors='coerce').to_numpy(dtype=float)
            difference = np.abs(left_values - right_values)
            finite = np.isfinite(difference)
            maximum = float(difference[finite].max()) if finite.any() else 0.0
            equal = bool(np.allclose(left_values, right_values, rtol=0.0, atol=float_atol, equal_nan=True))
            report['columns'][column] = {'equal': equal, 'max_abs_error': maximum}
        else:
            equal = bool(left.fillna('<NA>').astype(str).equals(right.fillna('<NA>').astype(str)))
            report['columns'][column] = {'equal': equal}
        exact = exact and equal
    report['passed'] = bool(exact)
    return report


def _validate_reference(registry, score_df, reference_raw_run_dir, float_atol):
    reference_dir = os.path.realpath(reference_raw_run_dir)
    reference_registry_dir = os.path.join(reference_dir, 'tailguard', 'pseudoclasses')
    actual_members = registry['members_df']
    actual_classes = registry['classes_df']
    actual_edges = registry['tail_edges_df']
    expected_members = pd.read_csv(os.path.join(reference_registry_dir, 'pseudo_class_members.csv'))
    expected_classes = pd.read_csv(os.path.join(reference_registry_dir, 'pseudo_classes.csv'))
    expected_edges = pd.read_csv(os.path.join(reference_registry_dir, 'tail_adaptive_neighborhood_edges.csv'))
    expected_scores = pd.read_csv(os.path.join(reference_dir, 'tailguard', 'memory', 'memory_eval_scores.csv'))
    actual_score_keys = _test_key_frame(score_df)
    expected_score_keys = _test_key_frame(expected_scores)
    comparisons = {
        'members': _frame_comparison(
            actual_members, expected_members, ['sample_idx'], float_atol, 'pseudo_class_members'
        ),
        'classes': _frame_comparison(
            actual_classes, expected_classes, ['pseudo_class_id'], float_atol, 'pseudo_classes'
        ),
        'edges': _frame_comparison(
            actual_edges,
            expected_edges,
            ['sample_idx_i', 'sample_idx_j'],
            max(float_atol, 2e-4),
            'tail_adaptive_neighborhood_edges',
        ),
        'scores': _frame_comparison(
            actual_score_keys.drop(columns=['img_path']),
            expected_score_keys.drop(columns=['img_path']),
            list(SCORE_KEY_COLUMNS),
            float_atol,
            'memory_eval_scores',
        ),
    }
    return {
        'reference_raw_run_dir': reference_dir,
        'float_atol': float(float_atol),
        'comparisons': comparisons,
        'passed': all(payload['passed'] for payload in comparisons.values()),
    }


def replay_coverage(parsed_args):
    started_at = _utc_now()
    output_dir = os.path.realpath(parsed_args.output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir already exists: {}'.format(output_dir))
    run_dir, artifact_paths, retained, removed = _load_full_artifacts(parsed_args.full_run_dir)
    checkpoint_path = os.path.realpath(
        parsed_args.encoder_checkpoint_path or os.path.join(run_dir, 'final_model.pt')
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'encoder checkpoint is missing; pass --encoder_checkpoint_path: {}'.format(checkpoint_path)
        )
    metadata = load_train_checkpoint_metadata(checkpoint_path)
    profile = _resolve_profile(parsed_args, metadata)
    data_path = os.path.realpath(parsed_args.data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError('data_path does not exist: {}'.format(data_path))
    full_members = pd.read_csv(artifact_paths['full_members'])
    _stable_identity_audit(retained, full_members)
    full_scores = pd.read_csv(artifact_paths['full_scores'])

    if not torch.cuda.is_available():
        raise RuntimeError('coverage replay requires CUDA')
    device = torch.device('cuda:{}'.format(int(parsed_args.gpu)))
    model, _, loaded_metadata = load_model_from_train_checkpoint(
        checkpoint_path, torch.device('cpu')
    )
    model = model.to(device).eval()
    transform, _ = get_data_transforms(metadata['image_size'], metadata['crop_size'])
    train_paths = _path_map_for_retained(retained, data_path, profile.item_list)
    test_paths = _path_map_for_test(full_scores, data_path)
    all_paths = list(train_paths.values()) + list(test_paths.values())
    cache_audit = ensure_feature_cache(
        model,
        all_paths,
        transform,
        os.path.realpath(parsed_args.feature_cache_dir),
        device,
        int(parsed_args.batch_size),
    )
    path_to_key = cache_audit.pop('path_to_key')
    cls_payload = _payload_from_cache(
        retained,
        train_paths,
        path_to_key,
        os.path.realpath(parsed_args.feature_cache_dir),
    )
    registry = _build_raw_registry(retained, removed, cls_payload)
    train_cache = _memory_cache_for_registry(
        registry,
        retained,
        train_paths,
        path_to_key,
        os.path.realpath(parsed_args.feature_cache_dir),
    )
    max_patches = int(_checkpoint_args_get(metadata['args'], 'tg_mem_max_patches_per_class', 20000))
    memory_system = build_memory_system_from_cache(
        _registry_wrapper(registry), train_cache, max_patches
    )
    del train_cache
    eval_args = SimpleNamespace(
        batch_size=int(parsed_args.batch_size),
        tg_memory_topk_ratio=float(parsed_args.tg_memory_topk_ratio),
        tg_memory_fusion_lambda=float(parsed_args.tg_memory_fusion_lambda),
        tg_memory_route_margin_threshold=float(parsed_args.tg_memory_route_margin_threshold),
        tg_memory_min_class_members=int(parsed_args.tg_memory_min_class_members),
        tg_mem_chunk_size=int(_checkpoint_args_get(metadata['args'], 'tg_mem_chunk_size', 4096)),
    )
    score_df = score_test_from_cache(
        full_scores,
        test_paths,
        path_to_key,
        os.path.realpath(parsed_args.feature_cache_dir),
        memory_system,
        eval_args,
        device,
    )
    per_class, macro_auroc = _macro_image_auroc(score_df)
    validation = None
    if parsed_args.reference_raw_run_dir:
        validation = _validate_reference(
            registry, score_df, parsed_args.reference_raw_run_dir, parsed_args.float_atol
        )

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    temporary = tempfile.mkdtemp(prefix=os.path.basename(output_dir) + '.inprogress-', dir=os.path.dirname(output_dir))
    try:
        registry['members_df'].to_csv(os.path.join(temporary, 'pseudo_class_members.csv'), index=False)
        registry['classes_df'].to_csv(os.path.join(temporary, 'pseudo_classes.csv'), index=False)
        registry['tail_edges_df'].to_csv(os.path.join(temporary, 'tail_adaptive_neighborhood_edges.csv'), index=False)
        score_df.to_csv(os.path.join(temporary, 'coverage_eval_scores.csv'), index=False)
        per_class.to_csv(os.path.join(temporary, 'coverage_per_class_metrics.csv'), index=False)
        summary = {
            'schema_version': SCHEMA_VERSION,
            'evaluator': 'tailguard_coverage_replay',
            'no_training': True,
            'started_at_utc': started_at,
            'finished_at_utc': _utc_now(),
            'full_run_dir': run_dir,
            'dataset_profile': profile.name,
            'data_path': data_path,
            'checkpoint_path': checkpoint_path,
            'checkpoint_sha256': _sha256_file(checkpoint_path),
            'checkpoint_iteration': loaded_metadata['iteration'],
            'feature_cache_dir': os.path.realpath(parsed_args.feature_cache_dir),
            'feature_cache_audit': cache_audit,
            'protocol': {
                'memory_topk_ratio': eval_args.tg_memory_topk_ratio,
                'memory_fusion_lambda': eval_args.tg_memory_fusion_lambda,
                'memory_route_margin_threshold': eval_args.tg_memory_route_margin_threshold,
                'memory_min_class_members': eval_args.tg_memory_min_class_members,
                'max_patches_per_class': max_patches,
            },
            'registry': registry['summary'],
            'coverage_I-AUROC': macro_auroc,
            'num_test_images': len(score_df),
            'validation': validation,
        }
        _write_json(os.path.join(temporary, 'coverage_replay_summary.json'), summary)
        if not parsed_args.no_save_memory_system:
            torch.save(memory_system, os.path.join(temporary, 'coverage_memory_system.pt'))
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    if validation is not None and not validation['passed']:
        raise RuntimeError('reference Raw-E validation failed; see {}'.format(
            os.path.join(output_dir, 'coverage_replay_summary.json')
        ))
    return summary
