"""Pseudo-class routing and tail memory evaluation for TailGuard."""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tailguard.method.memory_ops import (
    compute_memory_patch_scores,
    extract_encoder_patch_tokens,
    pool_patch_scores,
    resize_memory_map,
)
from tailguard.engine.ops import compute_image_level_scores, compute_pro, f1_score_max, get_gaussian_kernel, infer_anomaly_map_batch


def _replace_memory_entry(memory_system: Dict, pseudo_class_id: int, entry: Dict) -> Dict:
    replacement = dict(memory_system)
    replacement['class_entries'] = dict(memory_system['class_entries'])
    replacement['class_entries'][int(pseudo_class_id)] = entry
    return replacement


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), dim=-1)


def _extract_encoder_cls_embeddings(model, images: torch.Tensor) -> torch.Tensor:
    encoder = getattr(model, 'encoder', None)
    if encoder is None or not hasattr(encoder, 'prepare_tokens_with_masks'):
        raise ValueError('TailGuard pseudo-class memory requires a DINOv2-style encoder')
    tokens = encoder.prepare_tokens_with_masks(images)
    for block in encoder.blocks:
        tokens = block(tokens)
    return _normalize_rows(encoder.norm(tokens)[:, 0])


def _allocate_patch_quotas(capacities: List[int], max_patches: int) -> List[int]:
    total_patches = int(sum(capacities))
    target = total_patches if max_patches <= 0 else min(total_patches, int(max_patches))
    quotas = [0] * len(capacities)
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    remaining = target
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active = []
        for index in active:
            added = min(share, capacities[index] - quotas[index], remaining)
            quotas[index] += added
            remaining -= added
            if quotas[index] < capacities[index]:
                next_active.append(index)
            if remaining == 0:
                next_active.extend(active[active.index(index) + 1:])
                break
        active = next_active
    return quotas


def _sample_patch_features(features: torch.Tensor, quota: int) -> torch.Tensor:
    if quota <= 0:
        return features[:0]
    if quota >= features.shape[0]:
        return features
    indices = torch.linspace(0, features.shape[0] - 1, steps=quota).round().long()
    return features.index_select(0, indices)


def _registry_state(members_df: pd.DataFrame, classes_df: pd.DataFrame):
    required_members = {'sample_idx', 'pseudo_class_id'}
    required_classes = {'pseudo_class_id', 'pseudo_class_type', 'num_members'}
    missing_members = required_members.difference(members_df.columns)
    missing_classes = required_classes.difference(classes_df.columns)
    if missing_members:
        raise ValueError('pseudo-class members are missing: {}'.format(', '.join(sorted(missing_members))))
    if missing_classes:
        raise ValueError('pseudo-classes are missing: {}'.format(', '.join(sorted(missing_classes))))
    members = members_df.copy().reset_index(drop=True)
    classes = classes_df.copy().reset_index(drop=True)
    members['sample_idx'] = members['sample_idx'].astype(int)
    members['pseudo_class_id'] = members['pseudo_class_id'].astype(int)
    classes['pseudo_class_id'] = classes['pseudo_class_id'].astype(int)
    identity_columns = {'class_id', 'base_idx'}
    present_identity_columns = identity_columns.intersection(members.columns)
    if present_identity_columns and present_identity_columns != identity_columns:
        raise ValueError('pseudo-class members must include both class_id and base_idx')
    identity_mode = 'class_base' if present_identity_columns == identity_columns else 'legacy_sample_idx'
    if identity_mode == 'class_base':
        members['class_id'] = members['class_id'].astype(int)
        members['base_idx'] = members['base_idx'].astype(int)
        if members.duplicated(['class_id', 'base_idx']).any():
            raise ValueError('pseudo-class members must have unique class_id/base_idx identities')
    if members['sample_idx'].duplicated().any():
        raise ValueError('pseudo-class members must have unique sample_idx values')
    if classes['pseudo_class_id'].duplicated().any() or (classes['num_members'].astype(int) <= 0).any():
        raise ValueError('pseudo-class registry has invalid classes')
    counts = members.groupby('pseudo_class_id').size().to_dict()
    expected_counts = classes.set_index('pseudo_class_id')['num_members'].astype(int).to_dict()
    if counts != expected_counts:
        raise ValueError('pseudo-class membership counts do not match class registry')
    return members, classes, identity_mode


@torch.no_grad()
def build_tailguard_pseudoclass_memory_system(model,
                                                train_eval_dataloader,
                                                device,
                                                members_df: pd.DataFrame,
                                                classes_df: pd.DataFrame,
                                                args) -> Dict:
    members, classes, identity_mode = _registry_state(members_df, classes_df)
    class_type_lookup = classes.set_index('pseudo_class_id')['pseudo_class_type'].to_dict()
    if identity_mode == 'class_base':
        member_lookup = {
            (int(row.class_id), int(row.base_idx)): int(row.sample_idx)
            for row in members.itertuples(index=False)
        }
    else:
        member_lookup = {
            int(row.sample_idx): int(row.sample_idx)
            for row in members.itertuples(index=False)
        }
    members_by_sample_idx = members.set_index('sample_idx', drop=False)
    observed = {}
    loader_keys = set()
    was_training = model.training
    model.eval()
    try:
        for images, _, meta in train_eval_dataloader:
            images = images.to(device)
            patch_features, spatial_size = extract_encoder_patch_tokens(model, images)
            cls_embeddings = _extract_encoder_cls_embeddings(model, images)
            if identity_mode == 'class_base':
                batch_keys = [
                    (int(meta['class_id'][index]), int(meta['base_idx'][index]))
                    for index in range(images.shape[0])
                ]
            else:
                batch_keys = [int(meta['sample_idx'][index]) for index in range(images.shape[0])]
            for index, member_key in enumerate(batch_keys):
                if member_key in loader_keys:
                    raise ValueError('train-eval loader yielded a duplicate member identity')
                loader_keys.add(member_key)
                sample_idx = member_lookup.get(member_key)
                if sample_idx is None:
                    continue
                if member_key in observed:
                    raise ValueError('train-eval loader yielded a pseudo-class member more than once')
                pseudo_class_id = int(members_by_sample_idx.loc[sample_idx, 'pseudo_class_id'])
                record = {
                    'pseudo_class_id': pseudo_class_id,
                    'cls': cls_embeddings[index].detach().cpu().contiguous(),
                }
                if class_type_lookup[pseudo_class_id] == 'tail':
                    record.update({
                        'patches': patch_features[index].detach().cpu().contiguous(),
                        'spatial_size': tuple(int(value) for value in spatial_size),
                    })
                observed[member_key] = record
    finally:
        if was_training:
            model.train()

    expected_keys = set(member_lookup)
    if loader_keys != expected_keys:
        missing_from_loader = sorted(expected_keys.difference(loader_keys))
        unexpected_in_loader = sorted(loader_keys.difference(expected_keys))
        if identity_mode == 'legacy_sample_idx':
            raise ValueError(
                'legacy pseudo-class registry cannot be aligned to the rebuilt train-eval loader: '
                'registry sample_idx values differ from loader sample_idx values '
                '(registry={}, loader={}, missing={}, unexpected={}). Regenerate pseudo-class '
                'artifacts with class_id and base_idx before building final memory.'.format(
                    len(expected_keys), len(loader_keys), missing_from_loader[:10], unexpected_in_loader[:10]
                )
            )
        raise ValueError(
            'pseudo-class registry identities do not match the rebuilt train-eval loader '
            '(registry={}, loader={}, missing={}, unexpected={}).'.format(
                len(expected_keys), len(loader_keys), missing_from_loader[:10], unexpected_in_loader[:10]
            )
        )
    missing_members = sorted(expected_keys.difference(observed))
    if missing_members:
        raise ValueError('train-eval loader is missing pseudo-class members: {}'.format(missing_members[:10]))

    max_patches = int(getattr(args, 'tg_mem_max_patches_per_class', 20000))
    class_entries = {}
    for class_row in classes.sort_values('pseudo_class_id', kind='mergesort').itertuples(index=False):
        pseudo_class_id = int(class_row.pseudo_class_id)
        pseudo_class_type = str(class_row.pseudo_class_type)
        class_members = members.loc[members['pseudo_class_id'] == pseudo_class_id].sort_values(
            ['sample_idx'], kind='mergesort'
        )
        if identity_mode == 'class_base':
            class_member_keys = list(zip(
                class_members['class_id'].astype(int),
                class_members['base_idx'].astype(int),
            ))
        else:
            class_member_keys = class_members['sample_idx'].astype(int).tolist()
        records = [observed[member_key] for member_key in class_member_keys]
        cls_matrix = torch.stack([record['cls'] for record in records], dim=0)
        prototype = _normalize_rows(cls_matrix.mean(dim=0, keepdim=True))[0].cpu().contiguous()
        entry = {
            'pseudo_class_id': pseudo_class_id,
            'pseudo_class_type': pseudo_class_type,
            'num_members': int(len(records)),
            'member_sample_idx': class_members['sample_idx'].astype(int).tolist(),
            'prototype_cls': prototype,
            'member_cls': cls_matrix.cpu().contiguous(),
            'has_memory_bank': pseudo_class_type == 'tail',
            'num_patches': 0,
            'feature_dim': 0,
            'spatial_size': None,
            'member_patch_quotas': [],
        }
        if pseudo_class_type == 'tail':
            feature_dimensions = {int(record['patches'].shape[1]) for record in records}
            spatial_sizes = {record['spatial_size'] for record in records}
            if len(feature_dimensions) != 1 or len(spatial_sizes) != 1:
                raise ValueError('tail pseudo-class members have inconsistent patch shapes')
            capacities = [int(record['patches'].shape[0]) for record in records]
            quotas = _allocate_patch_quotas(capacities, max_patches)
            selected = [
                _sample_patch_features(record['patches'], quota)
                for record, quota in zip(records, quotas)
                if quota > 0
            ]
            if len(selected) == 0:
                raise ValueError('tail pseudo-class memory bank has no patch tokens')
            features = torch.cat(selected, dim=0).contiguous().cpu()
            entry.update({
                'features': features,
                'num_patches': int(features.shape[0]),
                'feature_dim': int(features.shape[1]),
                'spatial_size': next(iter(spatial_sizes)),
                'member_patch_quotas': [
                    {
                        'sample_idx': int(sample_idx),
                        'num_available_patches': int(capacity),
                        'num_selected_patches': int(quota),
                    }
                    for sample_idx, capacity, quota in zip(
                        class_members['sample_idx'].astype(int), capacities, quotas
                    )
                ],
            })
        elif pseudo_class_type != 'head':
            raise ValueError('unknown pseudo-class type: {}'.format(pseudo_class_type))
        class_entries[pseudo_class_id] = entry

    return {
        'schema_version': 1,
        'class_entries': class_entries,
        'num_pseudo_classes': int(len(class_entries)),
        'num_head_pseudo_classes': int(sum(entry['pseudo_class_type'] == 'head' for entry in class_entries.values())),
        'num_tail_pseudo_classes': int(sum(entry['pseudo_class_type'] == 'tail' for entry in class_entries.values())),
        'num_tail_memory_banks': int(sum(entry['has_memory_bank'] for entry in class_entries.values())),
        'max_patches_per_class': max_patches,
    }


def _routing_cache(memory_system: Dict, device):
    entries = memory_system.get('class_entries', {})
    if len(entries) == 0:
        raise ValueError('pseudo-class memory system has no classes')
    class_ids = sorted(int(class_id) for class_id in entries)
    prototypes = torch.stack([entries[class_id]['prototype_cls'] for class_id in class_ids], dim=0).to(device)
    return {
        'class_ids': class_ids,
        'prototypes': _normalize_rows(prototypes),
        'class_types': [str(entries[class_id]['pseudo_class_type']) for class_id in class_ids],
    }


def route_evidence_mask(mem_applied: torch.Tensor,
                        similarity_margin: torch.Tensor,
                        threshold: float) -> torch.Tensor:
    """Gate tail memory with an optional cosine-margin threshold."""
    threshold = float(threshold)
    if torch.isnan(torch.tensor(threshold)) or threshold == float('inf'):
        raise ValueError('tg_memory_route_margin_threshold must not be NaN or +inf')
    applied = mem_applied.to(dtype=torch.bool)
    if threshold == float('-inf'):
        return applied
    margins = similarity_margin.to(device=mem_applied.device)
    return applied & torch.isfinite(margins) & (margins >= threshold)


def leave_one_out_memory_system(memory_system: Dict, pseudo_class_id: int, sample_idx: int) -> Dict:
    """Remove one query's patches and CLS contribution from its own tail class."""
    pseudo_class_id = int(pseudo_class_id)
    sample_idx = int(sample_idx)
    entry = memory_system['class_entries'][pseudo_class_id]
    if entry['pseudo_class_type'] != 'tail':
        return memory_system
    member_ids = [int(value) for value in entry['member_sample_idx']]
    if sample_idx not in member_ids:
        return memory_system
    quotas = entry.get('member_patch_quotas', [])
    if len(member_ids) <= 1:
        return _replace_memory_entry(memory_system, pseudo_class_id, {
            **entry,
            'member_sample_idx': [],
            'member_cls': entry.get('member_cls', torch.empty((0, 0)))[:0],
            'member_patch_quotas': [],
            'num_members': 0,
            'features': entry['features'][:0],
            'num_patches': 0,
            'has_memory_bank': False,
        })
    if not quotas:
        raise ValueError('tail memory entry is missing member patch quotas')
    start = 0
    slices = []
    retained_ids = []
    retained_quotas = []
    for quota in quotas:
        count = int(quota['num_selected_patches'])
        end = start + count
        member_id = int(quota['sample_idx'])
        if member_id != sample_idx and count > 0:
            slices.append(entry['features'][start:end])
            retained_ids.append(member_id)
            retained_quotas.append(dict(quota))
        start = end
    if start != int(entry['features'].shape[0]):
        raise ValueError('tail memory quotas do not align with stored patch features')
    if not slices:
        raise ValueError('leave-one-out unexpectedly removed the complete multi-member memory bank')
    replacement_entry = dict(entry)
    replacement_entry.update({
        'member_sample_idx': retained_ids,
        'member_cls': entry['member_cls'][[
            position for position, value in enumerate(member_ids) if value != sample_idx
        ]].contiguous() if 'member_cls' in entry else None,
        'member_patch_quotas': retained_quotas,
        'num_members': len(retained_ids),
        'features': torch.cat(slices, dim=0).contiguous(),
        'num_patches': int(sum(int(row['num_selected_patches']) for row in retained_quotas)),
    })
    return _replace_memory_entry(memory_system, pseudo_class_id, replacement_entry)








@torch.no_grad()


@torch.no_grad()
def compute_pseudoclass_memory_scores_for_batch(model,
                                                images: torch.Tensor,
                                                memory_system: Dict,
                                                args,
                                                device,
                                                routing_cache=None,
                                                device_memory_cache=None,
                                                query_sample_idx=None,
                                                leave_one_out=False):
    images = images.to(device)
    patch_features, spatial_size = extract_encoder_patch_tokens(model, images)
    cls_embeddings = _extract_encoder_cls_embeddings(model, images)
    cache = _routing_cache(memory_system, device) if routing_cache is None else routing_cache
    similarities = cls_embeddings @ cache['prototypes'].T
    unavailable_positions = torch.zeros_like(similarities, dtype=torch.bool)
    if leave_one_out and query_sample_idx is not None:
        class_position = {int(class_id): position for position, class_id in enumerate(cache['class_ids'])}
        member_owner = {
            int(sample_idx): int(pseudo_class_id)
            for pseudo_class_id, entry in memory_system['class_entries'].items()
            for sample_idx in entry.get('member_sample_idx', [])
        }
        for index, sample_idx in enumerate(query_sample_idx):
            owner_id = member_owner.get(int(sample_idx))
            if owner_id is None:
                continue
            owner = memory_system['class_entries'][owner_id]
            member_ids = [int(value) for value in owner.get('member_sample_idx', [])]
            member_cls = owner.get('member_cls')
            if int(sample_idx) not in member_ids:
                continue
            if member_cls is None:
                raise ValueError('leave-one-out routing requires member CLS embeddings')
            if len(member_ids) <= 1:
                similarities[index, class_position[owner_id]] = float('-inf')
                unavailable_positions[index, class_position[owner_id]] = True
                continue
            retained = [position for position, value in enumerate(member_ids) if value != int(sample_idx)]
            loo_prototype = _normalize_rows(member_cls[retained].mean(dim=0, keepdim=True))[0].to(device)
            similarities[index, class_position[owner_id]] = cls_embeddings[index] @ loo_prototype
    best_positions = similarities.argmax(dim=1)
    best_similarity = similarities.gather(1, best_positions[:, None])[:, 0]
    if similarities.shape[1] > 1:
        second_similarity = torch.topk(similarities, k=2, dim=1).values[:, 1]
    else:
        second_similarity = torch.full_like(best_similarity, float('nan'))
    chunk_size = int(getattr(args, 'tg_mem_chunk_size', getattr(args, 'mem_chunk_size', 4096)))
    topk_ratio = float(getattr(args, 'tg_memory_topk_ratio', getattr(args, 'mem_topk_ratio', 0.05)))
    entries = memory_system['class_entries']
    min_class_members = int(getattr(args, 'tg_memory_min_class_members', 1))
    if min_class_members < 1:
        raise ValueError('tg_memory_min_class_members must be >= 1')
    scores = []
    maps = []
    mem_applied = []
    predicted_ids = []
    predicted_types = []
    for index, best_position in enumerate(best_positions.tolist()):
        if bool(unavailable_positions[index].all()) or not torch.isfinite(best_similarity[index]):
            predicted_ids.append(-1)
            predicted_types.append('unavailable')
            scores.append(torch.tensor(0.0, device=device))
            maps.append(torch.zeros(spatial_size, device=device))
            mem_applied.append(0)
            continue
        pseudo_class_id = int(cache['class_ids'][best_position])
        pseudo_class_type = cache['class_types'][best_position]
        predicted_ids.append(pseudo_class_id)
        predicted_types.append(pseudo_class_type)
        if pseudo_class_type == 'head':
            scores.append(torch.tensor(0.0, device=device))
            maps.append(torch.zeros(spatial_size, device=device))
            mem_applied.append(0)
            continue
        entry = entries[pseudo_class_id]
        if int(entry.get('num_members', 0)) < min_class_members:
            predicted_types[-1] = 'tail_unsupported'
            scores.append(torch.tensor(0.0, device=device))
            maps.append(torch.zeros(spatial_size, device=device))
            mem_applied.append(0)
            continue
        query_memory_system = memory_system
        if leave_one_out and query_sample_idx is not None:
            query_memory_system = leave_one_out_memory_system(
                memory_system,
                pseudo_class_id,
                int(query_sample_idx[index]),
            )
            entry = query_memory_system['class_entries'][pseudo_class_id]
        if not entry.get('has_memory_bank') or 'features' not in entry:
            raise ValueError('predicted tail pseudo-class has no memory bank: {}'.format(pseudo_class_id))
        memory_features = entry['features']
        if device_memory_cache is not None and query_memory_system is memory_system:
            memory_features = device_memory_cache.setdefault(
                pseudo_class_id,
                entry['features'].to(device),
            )
        patch_scores = compute_memory_patch_scores(
            patch_features[index],
            memory_features,
            chunk_size=chunk_size,
        )
        scores.append(pool_patch_scores(patch_scores, topk_ratio=topk_ratio))
        maps.append(patch_scores.reshape(spatial_size))
        mem_applied.append(1)
    return {
        'memory_scores': torch.stack(scores),
        'memory_patch_maps': torch.stack(maps),
        'mem_applied': torch.tensor(mem_applied, dtype=torch.int64),
        'predicted_pseudo_class_id': torch.tensor(predicted_ids, dtype=torch.int64),
        'predicted_pseudo_class_type': predicted_types,
        'top1_similarity': best_similarity.detach().cpu(),
        'second_similarity': second_similarity.detach().cpu(),
        'similarity_margin': (best_similarity - second_similarity).detach().cpu(),
    }


@torch.no_grad()
def run_pseudoclass_memory_evaluation(model,
                                      test_data_list,
                                      item_list,
                                      memory_system: Dict,
                                      args,
                                      device):
    was_training = model.training
    model.eval()
    try:
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
        rows: List[dict] = []
        lambda_value = float(getattr(args, 'tg_memory_fusion_lambda', 1.0))
        route_margin_threshold = float(getattr(args, 'tg_memory_route_margin_threshold', float('-inf')))
        batch_size = int(args.batch_size)
        num_workers = int(args.num_workers)
        max_ratio = float(getattr(args, 'diag_max_ratio', getattr(args, 'max_ratio', 0.01)))
        resize_mask = int(getattr(args, 'diag_resize_mask', getattr(args, 'resize_mask', 256)))
        metric_modes = ('recon', 'memory', 'fused')
        per_class_metrics_by_mode = {mode: [] for mode in metric_modes}
        routing_cache = _routing_cache(memory_system, device)
        device_memory_cache = {}

        for class_id, (class_name, test_data) in enumerate(zip(item_list, test_data_list)):
            loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            predictions = {
                mode: {'gt_sp': [], 'pred_sp': [], 'gt_px': [], 'pred_px': []}
                for mode in metric_modes
            }
            class_row_count = 0
            for images, gt, label, img_path in loader:
                images = images.to(device)
                gt = gt.to(device)
                recon_map, gt = infer_anomaly_map_batch(
                    model,
                    images,
                    gaussian_kernel,
                    resize_mask=resize_mask,
                    gt=gt,
                )
                recon_scores = compute_image_level_scores(recon_map, max_ratio=max_ratio)
                memory_result = compute_pseudoclass_memory_scores_for_batch(
                    model,
                    images,
                    memory_system,
                    args,
                    device,
                    routing_cache=routing_cache,
                    device_memory_cache=device_memory_cache,
                )
                memory_scores = memory_result['memory_scores']
                memory_maps = resize_memory_map(
                    memory_result['memory_patch_maps'].to(device), recon_map.shape[-2:]
                ).unsqueeze(1)
                route_eligible = memory_result['mem_applied'].to(device=device)
                similarity_margin = memory_result['similarity_margin'].to(device=device)
                tail_mask = route_evidence_mask(
                    route_eligible,
                    similarity_margin,
                    route_margin_threshold,
                )
                fused_scores = torch.where(tail_mask, recon_scores + lambda_value * memory_scores, recon_scores)
                fused_maps = torch.where(
                    tail_mask[:, None, None, None],
                    recon_map + lambda_value * memory_maps,
                    recon_map,
                )
                memory_mode_scores = torch.where(tail_mask, memory_scores, recon_scores)
                memory_mode_maps = torch.where(
                    tail_mask[:, None, None, None], memory_maps, recon_map
                )
                scores_by_mode = {'recon': recon_scores, 'memory': memory_mode_scores, 'fused': fused_scores}
                maps_by_mode = {'recon': recon_map, 'memory': memory_mode_maps, 'fused': fused_maps}

                gt_bool = gt.bool()
                if gt_bool.shape[1] > 1:
                    gt_bool = torch.max(gt_bool, dim=1, keepdim=True)[0]
                for mode in metric_modes:
                    predictions[mode]['gt_sp'].append(label.cpu())
                    predictions[mode]['pred_sp'].append(scores_by_mode[mode].detach().cpu())
                    predictions[mode]['gt_px'].append(gt_bool[:, 0].detach().cpu())
                    predictions[mode]['pred_px'].append(maps_by_mode[mode][:, 0].detach().cpu())
                for index in range(images.shape[0]):
                    rows.append({
                        'class_id': int(class_id),
                        'class_name': str(class_name),
                        'img_path': str(img_path[index]),
                        'label': int(label[index].item()),
                        'recon_score': float(recon_scores[index].item()),
                        'memory_score': float(memory_scores[index].item()),
                        'final_score': float(fused_scores[index].item()),
                        'route_eligible': int(memory_result['mem_applied'][index].item()),
                        'mem_applied': int(tail_mask[index].item()),
                        'predicted_pseudo_class_id': int(memory_result['predicted_pseudo_class_id'][index].item()),
                        'predicted_pseudo_class_type': memory_result['predicted_pseudo_class_type'][index],
                        'top1_similarity': float(memory_result['top1_similarity'][index].item()),
                        'second_similarity': float(memory_result['second_similarity'][index].item()),
                        'similarity_margin': float(memory_result['similarity_margin'][index].item()),
                    })
                    class_row_count += 1

            from sklearn.metrics import average_precision_score, roc_auc_score
            from tailguard.engine.ops import compute_pro, f1_score_max
            for mode in metric_modes:
                gt_sp_np = torch.cat(predictions[mode]['gt_sp']).flatten().numpy()
                pred_sp_np = torch.cat(predictions[mode]['pred_sp']).flatten().numpy()
                gt_px_np = torch.cat(predictions[mode]['gt_px']).numpy()
                pred_px_np = torch.cat(predictions[mode]['pred_px']).numpy()
                per_class_metrics_by_mode[mode].append({
                    'class_id': int(class_id),
                    'class_name': str(class_name),
                    'I-AUROC': float(roc_auc_score(gt_sp_np, pred_sp_np)),
                    'I-AP': float(average_precision_score(gt_sp_np, pred_sp_np)),
                    'I-F1': float(f1_score_max(gt_sp_np, pred_sp_np)),
                    'P-AUROC': float(roc_auc_score(gt_px_np.ravel(), pred_px_np.ravel())),
                    'P-AP': float(average_precision_score(gt_px_np.ravel(), pred_px_np.ravel())),
                    'P-F1': float(f1_score_max(gt_px_np.ravel(), pred_px_np.ravel())),
                    'P-AUPRO': float(compute_pro(gt_px_np, pred_px_np)),
                    'num_samples': int(class_row_count),
                })

        metrics_by_mode = {}
        for mode in metric_modes:
            metrics_df = pd.DataFrame(per_class_metrics_by_mode[mode])
            metrics_by_mode[mode] = {
                'summary': {
                    'I-AUROC': float(metrics_df['I-AUROC'].mean()),
                    'I-AP': float(metrics_df['I-AP'].mean()),
                    'I-F1': float(metrics_df['I-F1'].mean()),
                    'P-AUROC': float(metrics_df['P-AUROC'].mean()),
                    'P-AP': float(metrics_df['P-AP'].mean()),
                    'P-F1': float(metrics_df['P-F1'].mean()),
                    'P-AUPRO': float(metrics_df['P-AUPRO'].mean()),
                },
                'per_class_metrics': per_class_metrics_by_mode[mode],
            }
        score_df = pd.DataFrame(rows)
        summary = dict(metrics_by_mode['fused']['summary'])
        summary.update({
            'num_pseudo_classes': int(memory_system['num_pseudo_classes']),
            'num_head_pseudo_classes': int(memory_system['num_head_pseudo_classes']),
            'num_tail_pseudo_classes': int(memory_system['num_tail_pseudo_classes']),
            'num_tail_memory_banks': int(memory_system['num_tail_memory_banks']),
            'route_eligible_ratio': float(score_df['route_eligible'].mean()) if len(score_df) > 0 else 0.0,
            'memory_applied_ratio': float(score_df['mem_applied'].mean()) if len(score_df) > 0 else 0.0,
            'route_margin_threshold': route_margin_threshold,
            'memory_min_class_members': int(getattr(args, 'tg_memory_min_class_members', 1)),
        })
    finally:
        if was_training:
            model.train()
    return score_df, metrics_by_mode['fused']['per_class_metrics'], summary, metrics_by_mode
