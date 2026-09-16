"""Low-level patch-memory operations shared by CTE and coverage replay."""

import torch
import torch.nn.functional as F

from tailguard.engine.ops import compute_image_level_scores


def extract_encoder_patch_tokens(model, images):
    output = model(images, return_patch_tokens=True)
    if len(output) != 4:
        raise RuntimeError("model did not return patch tokens")
    _, _, patch_tokens, spatial_size = output
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    return patch_tokens, tuple(int(value) for value in spatial_size)


def compute_memory_patch_scores(test_features, memory_features, chunk_size):
    distances = []
    memory_features = memory_features.to(test_features.device)
    for start in range(0, test_features.shape[0], chunk_size):
        end = min(start + chunk_size, test_features.shape[0])
        query = test_features[start:end]
        nearest_similarity = (query @ memory_features.T).max(dim=1).values
        distances.append(1.0 - nearest_similarity)
    return torch.cat(distances, dim=0)


def pool_patch_scores(patch_scores, topk_ratio):
    return compute_image_level_scores(
        patch_scores.reshape(1, -1),
        max_ratio=float(topk_ratio),
    )[0]


def resize_memory_map(memory_map, output_size):
    resized = F.interpolate(
        memory_map.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized[:, 0]
