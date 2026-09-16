"""Pseudo-class registry construction for TailGuard."""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PSEUDO_CLASS_MEMBER_COLUMNS = [
    'sample_idx',
    'sample_key',
    'img_path',
    'class_id',
    'base_idx',
    'pseudo_class_id',
    'pseudo_class_type',
    'source_status',
    'head_group_id',
    'tail_component_id',
    'tail_component_size',
]

PSEUDO_CLASS_COLUMNS = [
    'pseudo_class_id',
    'pseudo_class_type',
    'head_group_id',
    'tail_component_id',
    'num_members',
    'num_h_clean',
    'num_t_head_normal',
    'num_t_open',
]

TAIL_EDGE_COLUMNS = [
    'sample_idx_i',
    'sample_idx_j',
    'cosine_similarity',
    'angular_distance_deg',
    'adaptive_angle_i',
    'adaptive_angle_j',
]


def _required_columns(frame: pd.DataFrame, columns: List[str], name: str):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError('{} is missing required columns: {}'.format(name, ', '.join(missing)))


def _sample_ids(frame: pd.DataFrame, name: str):
    _required_columns(frame, ['sample_idx'], name)
    sample_ids = frame['sample_idx'].astype(int)
    if sample_ids.duplicated().any():
        raise ValueError('{} contains duplicate sample_idx values'.format(name))
    return set(sample_ids.tolist())


def _embedding_lookup(cls_embeddings_payload: Dict):
    required = {'sample_idx', 'embeddings'}
    missing = required.difference(cls_embeddings_payload)
    if missing:
        raise ValueError('cls embeddings payload is missing: {}'.format(', '.join(sorted(missing))))
    sample_ids = [int(value) for value in cls_embeddings_payload['sample_idx']]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError('cls embeddings payload contains duplicate sample_idx values')
    embeddings = cls_embeddings_payload['embeddings'].float().cpu()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sample_ids):
        raise ValueError('cls embeddings payload has invalid embedding shape')
    if not torch.isfinite(embeddings).all():
        raise ValueError('cls embeddings payload contains non-finite values')
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if torch.any(norms <= 1e-12):
        raise ValueError('cls embeddings payload contains zero-norm vectors')
    embeddings = F.normalize(embeddings, dim=1)
    return {sample_idx: embeddings[index] for index, sample_idx in enumerate(sample_ids)}


def _components_from_edges(sample_ids: List[int], edges: List[tuple]):
    adjacency = {sample_idx: set() for sample_idx in sample_ids}
    for sample_idx_i, sample_idx_j in edges:
        adjacency[sample_idx_i].add(sample_idx_j)
        adjacency[sample_idx_j].add(sample_idx_i)
    components = []
    visited = set()
    for sample_idx in sorted(sample_ids):
        if sample_idx in visited:
            continue
        component = []
        stack = [sample_idx]
        visited.add(sample_idx)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _member_row(row: pd.Series,
                pseudo_class_id: int,
                pseudo_class_type: str,
                source_status: str,
                head_group_id: int,
                tail_component_id: int,
                tail_component_size: int):
    return {
        'sample_idx': int(row['sample_idx']),
        'sample_key': int(row['sample_key']),
        'img_path': str(row['img_path']),
        'class_id': int(row['class_id']),
        'base_idx': int(row['base_idx']),
        'pseudo_class_id': int(pseudo_class_id),
        'pseudo_class_type': pseudo_class_type,
        'source_status': source_status,
        'head_group_id': int(head_group_id),
        'tail_component_id': int(tail_component_id),
        'tail_component_size': int(tail_component_size),
    }


def build_tailguard_pseudoclass_registry(h_clean_df: pd.DataFrame,
                                            tail_head_normal_df: pd.DataFrame,
                                            tail_open_df: pd.DataFrame,
                                            retained_samples_df: pd.DataFrame,
                                            removed_samples_df: pd.DataFrame,
                                            cls_embeddings_payload: Dict):
    h_clean_df = h_clean_df.copy().reset_index(drop=True)
    tail_head_normal_df = tail_head_normal_df.copy().reset_index(drop=True)
    tail_open_df = tail_open_df.copy().reset_index(drop=True)
    retained_samples_df = retained_samples_df.copy().reset_index(drop=True)
    removed_samples_df = removed_samples_df.copy().reset_index(drop=True)

    identity_columns = ['class_id', 'base_idx']
    for name, frame, columns in [
        ('h_clean_samples', h_clean_df, ['sample_idx', 'sample_key', 'img_path', 'group_id'] + identity_columns),
        ('tail_head_normal_samples', tail_head_normal_df, ['sample_idx', 'sample_key', 'img_path', 'best_group_id'] + identity_columns),
        ('tail_open_samples', tail_open_df, ['sample_idx', 'sample_key', 'img_path', 'adaptive_angle'] + identity_columns),
        ('stage2_retained_samples', retained_samples_df, ['sample_idx'] + identity_columns),
        ('stage2_removed_samples', removed_samples_df, ['sample_idx']),
    ]:
        _required_columns(frame, columns, name)

    h_clean_ids = _sample_ids(h_clean_df, 'h_clean_samples')
    tail_head_normal_ids = _sample_ids(tail_head_normal_df, 'tail_head_normal_samples')
    tail_open_ids = _sample_ids(tail_open_df, 'tail_open_samples')
    retained_ids = _sample_ids(retained_samples_df, 'stage2_retained_samples')
    removed_ids = _sample_ids(removed_samples_df, 'stage2_removed_samples')
    for frame in [h_clean_df, tail_head_normal_df, tail_open_df, retained_samples_df]:
        frame['class_id'] = frame['class_id'].astype(int)
        frame['base_idx'] = frame['base_idx'].astype(int)
    if retained_samples_df.duplicated(['class_id', 'base_idx']).any():
        raise ValueError('stage2_retained_samples contains duplicate class_id/base_idx identities')
    retained_identity_by_sample = retained_samples_df.set_index('sample_idx')[['class_id', 'base_idx']]
    for name, frame in [
        ('h_clean_samples', h_clean_df),
        ('tail_head_normal_samples', tail_head_normal_df),
        ('tail_open_samples', tail_open_df),
    ]:
        identities = frame.set_index('sample_idx')[['class_id', 'base_idx']]
        missing_identity_ids = identities.index.difference(retained_identity_by_sample.index)
        if len(missing_identity_ids) > 0:
            raise ValueError('{} contains samples missing from stage2_retained_samples'.format(name))
        expected = retained_identity_by_sample.loc[identities.index]
        if not identities.equals(expected):
            raise ValueError('{} identity does not match stage2_retained_samples'.format(name))

    source_ids = h_clean_ids | tail_head_normal_ids | tail_open_ids
    source_overlap = (
        (h_clean_ids & tail_head_normal_ids)
        | (h_clean_ids & tail_open_ids)
        | (tail_head_normal_ids & tail_open_ids)
    )
    if source_overlap:
        raise ValueError('pseudo-class source partitions overlap on sample_idx')
    if retained_ids & removed_ids:
        raise ValueError('stage2 retained and removed samples overlap on sample_idx')
    if source_ids != retained_ids:
        raise ValueError('pseudo-class source partitions do not exactly cover Stage 2 retained samples')
    if source_ids & removed_ids:
        raise ValueError('removed samples appear in pseudo-class source partitions')

    h_clean_df['group_id'] = h_clean_df['group_id'].astype(int)
    if (h_clean_df['group_id'] < 0).any():
        raise ValueError('h_clean_samples contains invalid group_id values')
    head_group_ids = sorted(h_clean_df['group_id'].unique().tolist())
    head_group_to_pseudo_id = {
        int(group_id): pseudo_class_id
        for pseudo_class_id, group_id in enumerate(head_group_ids)
    }

    tail_head_normal_df['best_group_id'] = tail_head_normal_df['best_group_id'].astype(int)
    invalid_normal_groups = sorted(
        set(tail_head_normal_df['best_group_id'].tolist()).difference(head_group_to_pseudo_id)
    )
    if invalid_normal_groups:
        raise ValueError('tail_head_normal_samples reference missing H_clean groups: {}'.format(invalid_normal_groups))

    embeddings = _embedding_lookup(cls_embeddings_payload)
    missing_tail_embeddings = sorted(tail_open_ids.difference(embeddings))
    if missing_tail_embeddings:
        raise ValueError('cls embeddings are missing T_open samples: {}'.format(missing_tail_embeddings[:10]))

    tail_edges = []
    tail_edge_pairs = []
    tail_component_by_sample = {}
    tail_component_sizes = {}
    if len(tail_open_df) > 0:
        tail_open_df['sample_idx'] = tail_open_df['sample_idx'].astype(int)
        tail_open_df['adaptive_angle'] = tail_open_df['adaptive_angle'].astype(float)
        if (~np.isfinite(tail_open_df['adaptive_angle'].to_numpy())).any() or (tail_open_df['adaptive_angle'] < 0).any():
            raise ValueError('tail_open_samples contains invalid adaptive_angle values')
        ordered_tail_df = tail_open_df.sort_values('sample_idx', kind='mergesort').reset_index(drop=True)
        ordered_tail_ids = ordered_tail_df['sample_idx'].astype(int).tolist()
        adaptive_angles = ordered_tail_df['adaptive_angle'].to_numpy(dtype=float)
        matrix = torch.stack([embeddings[sample_idx] for sample_idx in ordered_tail_ids], dim=0)
        similarities = torch.clamp(matrix @ matrix.T, min=-1.0, max=1.0)
        angles = torch.rad2deg(torch.acos(similarities)).numpy()
        for index_i, sample_idx_i in enumerate(ordered_tail_ids):
            for index_j in range(index_i + 1, len(ordered_tail_ids)):
                angle = float(angles[index_i, index_j])
                if angle <= adaptive_angles[index_i] and angle <= adaptive_angles[index_j]:
                    sample_idx_j = ordered_tail_ids[index_j]
                    tail_edge_pairs.append((sample_idx_i, sample_idx_j))
                    tail_edges.append({
                        'sample_idx_i': int(sample_idx_i),
                        'sample_idx_j': int(sample_idx_j),
                        'cosine_similarity': float(similarities[index_i, index_j].item()),
                        'angular_distance_deg': angle,
                        'adaptive_angle_i': float(adaptive_angles[index_i]),
                        'adaptive_angle_j': float(adaptive_angles[index_j]),
                    })
        components = _components_from_edges(ordered_tail_ids, tail_edge_pairs)
        for component_id, component in enumerate(components):
            tail_component_sizes[component_id] = len(component)
            for sample_idx in component:
                tail_component_by_sample[sample_idx] = component_id
    else:
        components = []

    if len(tail_open_ids) > 0 and len(tail_component_by_sample) != len(tail_open_ids):
        raise ValueError('tail pseudo-class components do not cover T_open samples')

    member_rows = []
    class_rows = []
    for group_id in head_group_ids:
        pseudo_class_id = head_group_to_pseudo_id[int(group_id)]
        group_h_clean = h_clean_df.loc[h_clean_df['group_id'] == group_id].sort_values(
            'sample_idx', kind='mergesort'
        )
        group_tail_normal = tail_head_normal_df.loc[
            tail_head_normal_df['best_group_id'] == group_id
        ].sort_values('sample_idx', kind='mergesort')
        for _, row in group_h_clean.iterrows():
            member_rows.append(_member_row(row, pseudo_class_id, 'head', 'H_clean', group_id, -1, 0))
        for _, row in group_tail_normal.iterrows():
            member_rows.append(_member_row(row, pseudo_class_id, 'head', 'T_head_normal', group_id, -1, 0))
        class_rows.append({
            'pseudo_class_id': int(pseudo_class_id),
            'pseudo_class_type': 'head',
            'head_group_id': int(group_id),
            'tail_component_id': -1,
            'num_members': int(len(group_h_clean) + len(group_tail_normal)),
            'num_h_clean': int(len(group_h_clean)),
            'num_t_head_normal': int(len(group_tail_normal)),
            'num_t_open': 0,
        })

    tail_by_sample = tail_open_df.set_index('sample_idx', drop=False)
    tail_pseudo_start = len(head_group_ids)
    for component_id, component in enumerate(components):
        pseudo_class_id = tail_pseudo_start + component_id
        component_size = tail_component_sizes[component_id]
        for sample_idx in component:
            member_rows.append(_member_row(
                tail_by_sample.loc[sample_idx],
                pseudo_class_id,
                'tail',
                'T_open',
                -1,
                component_id,
                component_size,
            ))
        class_rows.append({
            'pseudo_class_id': int(pseudo_class_id),
            'pseudo_class_type': 'tail',
            'head_group_id': -1,
            'tail_component_id': int(component_id),
            'num_members': int(component_size),
            'num_h_clean': 0,
            'num_t_head_normal': 0,
            'num_t_open': int(component_size),
        })

    members_df = pd.DataFrame(member_rows, columns=PSEUDO_CLASS_MEMBER_COLUMNS)
    classes_df = pd.DataFrame(class_rows, columns=PSEUDO_CLASS_COLUMNS)
    tail_edges_df = pd.DataFrame(tail_edges, columns=TAIL_EDGE_COLUMNS)
    member_ids = _sample_ids(members_df, 'pseudo_class_members') if len(members_df) > 0 else set()
    if member_ids != retained_ids:
        raise ValueError('pseudo-class registry does not exactly cover Stage 2 retained samples')
    if members_df.duplicated(['class_id', 'base_idx']).any():
        raise ValueError('pseudo-class registry contains duplicate class_id/base_idx identities')
    if len(classes_df) > 0 and (classes_df['num_members'].astype(int) <= 0).any():
        raise ValueError('pseudo-class registry contains an empty class')
    if len(components) != int((classes_df['pseudo_class_type'] == 'tail').sum()):
        raise ValueError('tail pseudo-class registry has inconsistent component count')

    summary = {
        'status': 'completed',
        'cls_embedding_source': cls_embeddings_payload.get('embedding_source'),
        'num_retained_samples': int(len(retained_ids)),
        'num_removed_samples': int(len(removed_ids)),
        'num_pseudo_classes': int(len(classes_df)),
        'num_head_pseudo_classes': int(len(head_group_ids)),
        'num_tail_pseudo_classes': int(len(components)),
        'num_tail_graph_edges': int(len(tail_edges_df)),
        'num_tail_singleton_classes': int(sum(len(component) == 1 for component in components)),
        'head_group_to_pseudo_class_id': head_group_to_pseudo_id,
        'integrity': {
            'source_partitions_disjoint': True,
            'retained_removed_disjoint': True,
            'source_partitions_cover_retained': True,
            'removed_absent_from_registry': True,
            'registry_member_ids_unique': True,
            'registry_member_identities_unique': True,
            'source_identities_match_retained': True,
            'registry_covers_retained': True,
            'tail_components_cover_t_open': len(tail_component_by_sample) == len(tail_open_ids),
            'tail_classes_nonempty': not (len(classes_df) > 0 and (classes_df['num_members'].astype(int) <= 0).any()),
        },
    }
    return {
        'members_df': members_df,
        'classes_df': classes_df,
        'tail_edges_df': tail_edges_df,
        'summary': summary,
    }
