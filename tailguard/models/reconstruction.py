"""Dinomaly reconstruction model used by TailGuard."""

import math

import torch
import torch.nn as nn


class ViTill(nn.Module):
    def __init__(
        self,
        encoder,
        bottleneck,
        decoder,
        target_layers=None,
        fuse_layer_encoder=None,
        fuse_layer_decoder=None,
        mask_neighbor_size=0,
        remove_class_token=False,
        encoder_require_grad_layer=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers or [2, 3, 4, 5, 6, 7, 8, 9]
        self.fuse_layer_encoder = fuse_layer_encoder or [[0, 1, 2, 3, 4, 5, 6, 7]]
        self.fuse_layer_decoder = fuse_layer_decoder or [[0, 1, 2, 3, 4, 5, 6, 7]]
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer or []
        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, images, return_patch_tokens=False):
        tokens = self.encoder.prepare_tokens(images)
        encoder_layers = []
        for layer_index, block in enumerate(self.encoder.blocks):
            if layer_index > self.target_layers[-1]:
                continue
            if layer_index in self.encoder_require_grad_layer:
                tokens = block(tokens)
            else:
                with torch.no_grad():
                    tokens = block(tokens)
            if layer_index in self.target_layers:
                encoder_layers.append(tokens)

        side = int(math.sqrt(
            encoder_layers[0].shape[1] - 1 - self.encoder.num_register_tokens
        ))
        if self.remove_class_token:
            encoder_layers = [
                features[:, 1 + self.encoder.num_register_tokens :, :]
                for features in encoder_layers
            ]

        patch_tokens = self._fuse_features(encoder_layers)
        decoded = patch_tokens
        for block in self.bottleneck:
            decoded = block(decoded)

        attention_mask = (
            self._neighbor_mask(side, decoded.device)
            if self.mask_neighbor_size > 0
            else None
        )
        decoder_layers = []
        for block in self.decoder:
            decoded = block(decoded, attn_mask=attention_mask)
            decoder_layers.append(decoded)
        decoder_layers.reverse()

        encoder_outputs = [
            self._fuse_features([encoder_layers[index] for index in indices])
            for indices in self.fuse_layer_encoder
        ]
        decoder_outputs = [
            self._fuse_features([decoder_layers[index] for index in indices])
            for indices in self.fuse_layer_decoder
        ]

        if not self.remove_class_token:
            token_start = 1 + self.encoder.num_register_tokens
            encoder_outputs = [features[:, token_start:, :] for features in encoder_outputs]
            decoder_outputs = [features[:, token_start:, :] for features in decoder_outputs]
            patch_tokens = patch_tokens[:, token_start:, :]

        batch_size = images.shape[0]
        encoder_outputs = [
            features.permute(0, 2, 1).reshape(batch_size, -1, side, side).contiguous()
            for features in encoder_outputs
        ]
        decoder_outputs = [
            features.permute(0, 2, 1).reshape(batch_size, -1, side, side).contiguous()
            for features in decoder_outputs
        ]
        if return_patch_tokens:
            return encoder_outputs, decoder_outputs, patch_tokens.contiguous(), (side, side)
        return encoder_outputs, decoder_outputs

    @staticmethod
    def _fuse_features(features):
        return torch.stack(features, dim=1).mean(dim=1)

    def _neighbor_mask(self, feature_size, device):
        height = width = feature_size
        mask_height = mask_width = self.mask_neighbor_size
        mask = torch.ones(height, width, height, width, device=device)
        for row in range(height):
            for column in range(width):
                row_start = max(row - mask_height // 2, 0)
                row_end = min(row + mask_height // 2 + 1, height)
                column_start = max(column - mask_width // 2, 0)
                column_end = min(column + mask_width // 2 + 1, width)
                mask[row, column, row_start:row_end, column_start:column_end] = 0
        mask = mask.view(height * width, height * width)
        if self.remove_class_token:
            return mask
        prefix = 1 + self.encoder.num_register_tokens
        full_mask = torch.ones(height * width + prefix, width * height + prefix, device=device)
        full_mask[prefix:, prefix:] = mask
        return full_mask
