"""Shared feature and memory utilities for TailGuard evaluation replay."""

import math

import torch
import torch.nn.functional as F

from tailguard.method.memory_ops import compute_memory_patch_scores, pool_patch_scores


def _normalize_rows(tensor):
    return F.normalize(tensor.float(), dim=-1)


def encoder_features_once(model, images):
    """Extract Dinomaly patch features and final DINOv2 CLS in one pass."""
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "prepare_tokens"):
        raise ValueError("evaluation replay requires a ViTill model with a DINOv2 encoder")
    tokens = encoder.prepare_tokens(images)
    captured = []
    target_layers = set(int(value) for value in model.target_layers)
    for index, block in enumerate(encoder.blocks):
        tokens = block(tokens)
        if index in target_layers:
            captured.append(tokens)
    if len(captured) != len(model.target_layers):
        raise ValueError("model target layers were not all observed during the encoder pass")
    cls_embeddings = _normalize_rows(encoder.norm(tokens)[:, 0])
    side = int(math.sqrt(captured[0].shape[1] - 1 - encoder.num_register_tokens))
    reconstruction_tokens = captured
    if model.remove_class_token:
        reconstruction_tokens = [
            value[:, 1 + encoder.num_register_tokens :, :] for value in captured
        ]
    patch_tokens = model.fuse_feature(reconstruction_tokens)
    if not model.remove_class_token:
        patch_tokens = patch_tokens[:, 1 + encoder.num_register_tokens :, :]
    return (
        _normalize_rows(patch_tokens),
        (side, side),
        cls_embeddings,
        reconstruction_tokens,
    )


def _allocate_patch_quotas(capacities, max_patches):
    total = int(sum(capacities))
    target = total if max_patches <= 0 else min(total, int(max_patches))
    quotas = [0] * len(capacities)
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    remaining = target
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active = []
        for position, index in enumerate(active):
            added = min(share, capacities[index] - quotas[index], remaining)
            quotas[index] += added
            remaining -= added
            if quotas[index] < capacities[index]:
                next_active.append(index)
            if remaining == 0:
                next_active.extend(active[position + 1 :])
                break
        active = next_active
    return quotas


def _sample_patches(features, quota):
    if quota <= 0:
        return features[:0]
    if quota >= features.shape[0]:
        return features
    indices = torch.linspace(0, features.shape[0] - 1, steps=quota).round().long()
    return features.index_select(0, indices)


def build_memory_system_from_cache(registry, train_cache, max_patches):
    members = registry["members"]
    classes = registry["classes"]
    entries = {}
    for class_row in classes.itertuples(index=False):
        pseudo_class_id = int(class_row.pseudo_class_id)
        pseudo_class_type = str(class_row.pseudo_class_type)
        class_members = members.loc[
            members["pseudo_class_id"] == pseudo_class_id
        ].sort_values("sample_idx", kind="mergesort")
        keys = [
            (int(row.class_id), int(row.base_idx))
            for row in class_members.itertuples(index=False)
        ]
        records = [train_cache[key] for key in keys]
        cls_matrix = torch.stack([record["cls"] for record in records], dim=0)
        entry = {
            "pseudo_class_id": pseudo_class_id,
            "pseudo_class_type": pseudo_class_type,
            "num_members": len(records),
            "member_sample_idx": class_members["sample_idx"].astype(int).tolist(),
            "prototype_cls": _normalize_rows(cls_matrix.mean(dim=0, keepdim=True))[0]
            .cpu()
            .contiguous(),
            "member_cls": cls_matrix.cpu().contiguous(),
            "has_memory_bank": pseudo_class_type == "tail",
            "num_patches": 0,
            "feature_dim": 0,
            "spatial_size": None,
            "member_patch_quotas": [],
        }
        if pseudo_class_type == "tail":
            dimensions = {int(record["patches"].shape[1]) for record in records}
            spatial_sizes = {record["spatial_size"] for record in records}
            if len(dimensions) != 1 or len(spatial_sizes) != 1:
                raise ValueError(
                    "{} tail class {} has inconsistent patch shapes".format(
                        registry["variant"], pseudo_class_id
                    )
                )
            capacities = [int(record["patches"].shape[0]) for record in records]
            quotas = _allocate_patch_quotas(capacities, max_patches)
            selected = [
                _sample_patches(record["patches"], quota)
                for record, quota in zip(records, quotas)
                if quota > 0
            ]
            if not selected:
                raise ValueError(
                    "{} tail class {} has no memory patches".format(
                        registry["variant"], pseudo_class_id
                    )
                )
            features = torch.cat(selected, dim=0).contiguous().cpu()
            entry.update(
                {
                    "features": features,
                    "num_patches": int(features.shape[0]),
                    "feature_dim": int(features.shape[1]),
                    "spatial_size": next(iter(spatial_sizes)),
                    "member_patch_quotas": [
                        {
                            "sample_idx": int(sample_idx),
                            "num_available_patches": int(capacity),
                            "num_selected_patches": int(quota),
                        }
                        for sample_idx, capacity, quota in zip(
                            class_members["sample_idx"].astype(int), capacities, quotas
                        )
                    ],
                }
            )
        elif pseudo_class_type != "head":
            raise ValueError("unknown pseudo-class type: {}".format(pseudo_class_type))
        entries[pseudo_class_id] = entry
    return {
        "schema_version": 1,
        "class_entries": entries,
        "num_pseudo_classes": len(entries),
        "num_head_pseudo_classes": sum(
            row["pseudo_class_type"] == "head" for row in entries.values()
        ),
        "num_tail_pseudo_classes": sum(
            row["pseudo_class_type"] == "tail" for row in entries.values()
        ),
        "num_tail_memory_banks": sum(row["has_memory_bank"] for row in entries.values()),
        "max_patches_per_class": int(max_patches),
    }


def score_cached_features(
    patch_features,
    spatial_size,
    cls_embeddings,
    memory_system,
    args,
    device,
    routing_cache,
    device_memory_cache,
):
    similarities = cls_embeddings @ routing_cache["prototypes"].T
    best_positions = similarities.argmax(dim=1)
    best_similarity = similarities.gather(1, best_positions[:, None])[:, 0]
    if similarities.shape[1] > 1:
        second_similarity = torch.topk(similarities, k=2, dim=1).values[:, 1]
    else:
        second_similarity = torch.full_like(best_similarity, float("nan"))
    scores, maps, eligible, predicted_ids, predicted_types = [], [], [], [], []
    for index, position in enumerate(best_positions.tolist()):
        pseudo_class_id = int(routing_cache["class_ids"][position])
        pseudo_class_type = routing_cache["class_types"][position]
        entry = memory_system["class_entries"][pseudo_class_id]
        predicted_ids.append(pseudo_class_id)
        predicted_types.append(pseudo_class_type)
        supported_tail = (
            pseudo_class_type == "tail"
            and int(entry["num_members"]) >= int(args.tg_memory_min_class_members)
        )
        if not supported_tail:
            if pseudo_class_type == "tail":
                predicted_types[-1] = "tail_unsupported"
            scores.append(torch.tensor(0.0, device=device))
            maps.append(torch.zeros(spatial_size, device=device))
            eligible.append(0)
            continue
        memory_features = device_memory_cache.setdefault(
            pseudo_class_id, entry["features"].to(device)
        )
        patch_scores = compute_memory_patch_scores(
            patch_features[index],
            memory_features,
            chunk_size=int(args.tg_mem_chunk_size),
        )
        scores.append(pool_patch_scores(patch_scores, topk_ratio=args.tg_memory_topk_ratio))
        maps.append(patch_scores.reshape(spatial_size))
        eligible.append(1)
    return {
        "memory_scores": torch.stack(scores),
        "memory_patch_maps": torch.stack(maps),
        "route_eligible": torch.tensor(eligible, dtype=torch.int64, device=device),
        "predicted_pseudo_class_id": torch.tensor(predicted_ids, dtype=torch.int64),
        "predicted_pseudo_class_type": predicted_types,
        "top1_similarity": best_similarity.detach().cpu(),
        "second_similarity": second_similarity.detach().cpu(),
        "similarity_margin": (best_similarity - second_similarity).detach().cpu(),
    }
