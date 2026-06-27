from typing import Dict, List, Sequence

import torch

from dinov2.hub.backbones import dinov2_vitb14, dinov2_vitb14_reg, dinov2_vitl14, dinov2_vitl14_reg


def load_dinov2_model(config):
    model_name = str(config.encoder.model_name)
    use_registers = bool(config.encoder.use_registers)
    constructors = {
        'dinov2_vitb14': dinov2_vitb14_reg if use_registers else dinov2_vitb14,
        'dinov2_vitl14': dinov2_vitl14_reg if use_registers else dinov2_vitl14,
    }
    if model_name not in constructors:
        raise ValueError('unsupported model_name: {}'.format(model_name))
    model = constructors[model_name](pretrained=bool(config.encoder.pretrained), img_size=int(config.encoder.input_size))
    model.eval()
    return model


def resolve_layer_indices(num_blocks: int, high_layer: int, mid_layers: Sequence[int]) -> Dict[str, List[int]]:
    def resolve(index: int) -> int:
        return int(index if index >= 0 else num_blocks + index)
    resolved_high = resolve(int(high_layer))
    resolved_mid = [resolve(int(index)) for index in mid_layers]
    all_indices = sorted(set([resolved_high] + resolved_mid))
    return {
        'high': [resolved_high],
        'mid': resolved_mid,
        'all': all_indices,
    }


def _split_tokens(tokens: torch.Tensor, num_register_tokens: int):
    cls_tokens = tokens[:, 0, :]
    patch_tokens = tokens[:, 1 + int(num_register_tokens):, :]
    side = int(patch_tokens.shape[1] ** 0.5)
    if side * side != patch_tokens.shape[1]:
        raise ValueError('patch token count is not square: {}'.format(int(patch_tokens.shape[1])))
    patch_tokens = patch_tokens.reshape(tokens.shape[0], side, side, patch_tokens.shape[-1])
    return cls_tokens, patch_tokens


@torch.no_grad()
def extract_view_tokens(model, images: torch.Tensor, layer_config: Dict[str, List[int]], autocast_enabled: bool = False):
    num_register_tokens = int(getattr(model, 'num_register_tokens', 0))
    x = model.prepare_tokens(images)
    collected = {}
    autocast_device = 'cuda' if images.is_cuda else 'cpu'
    with torch.autocast(device_type=autocast_device, enabled=bool(autocast_enabled)):
        for block_index, block in enumerate(model.blocks):
            x = block(x)
            if block_index in layer_config['all']:
                collected[int(block_index)] = model.norm(x)
    high_tokens = collected[layer_config['high'][0]]
    mid_tokens = torch.stack([collected[index] for index in layer_config['mid']], dim=0).mean(dim=0)
    cls_tokens, patch_tokens_high = _split_tokens(high_tokens, num_register_tokens=num_register_tokens)
    _, patch_tokens_mid = _split_tokens(mid_tokens, num_register_tokens=num_register_tokens)
    return {
        'cls_tokens': cls_tokens.float(),
        'patch_tokens_high': patch_tokens_high.float(),
        'patch_tokens_mid': patch_tokens_mid.float(),
    }
