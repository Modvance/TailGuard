"""DINOv2 encoder loader for the TailGuard paper configuration."""

import logging
import os

import torch
from torch.hub import HASH_REGEX, download_url_to_file, urlparse

from tailguard.vendor.dinov2.models import vision_transformer as vision_transformer_dinov2


_LOGGER = logging.getLogger(__name__)
_WEIGHTS_DIR = "backbones/weights"
_ENCODER_NAME = "dinov2reg_vit_base_14"
_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
    "dinov2_vitb14_reg4_pretrain.pth"
)


def load(name, pretrained=True):
    """Build the frozen DINOv2 ViT-B/14 encoder used by TailGuard."""
    if name != _ENCODER_NAME:
        raise ValueError(
            "TailGuard supports only {!r}; received {!r}".format(_ENCODER_NAME, name)
        )
    model = vision_transformer_dinov2.vit_base(
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-8,
        num_register_tokens=4,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    if pretrained:
        checkpoint_path = download_cached_file(_CHECKPOINT_URL)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model


def download_cached_file(url, check_hash=True, progress=True):
    """Download an encoder checkpoint once and return its cached path."""
    os.makedirs(_WEIGHTS_DIR, exist_ok=True)
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
        filename = os.path.basename(urlparse(url).path)
    cached_file = os.path.join(_WEIGHTS_DIR, filename)
    if not os.path.exists(cached_file):
        _LOGGER.info('Downloading: "%s" to %s', url, cached_file)
        hash_prefix = None
        if check_hash:
            matched_hash = HASH_REGEX.search(filename)
            hash_prefix = matched_hash.group(1) if matched_hash else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file
