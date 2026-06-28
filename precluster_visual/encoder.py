from contextlib import nullcontext
from typing import List

import torch
import torch.nn.functional as F


class SiglipImageEncoder:
    def __init__(self, model_name: str, device: str = 'cuda', precision: str = 'fp16'):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError('transformers is required for SigLIP preclustering') from exc

        requested_device = str(device)
        if requested_device.startswith('cuda') and not torch.cuda.is_available():
            requested_device = 'cpu'
        self.device = torch.device(requested_device)
        self.precision = str(precision).lower()
        self.autocast_dtype = self._resolve_autocast_dtype(self.precision)
        model_kwargs = {}
        if self.device.type == 'cuda' and self.autocast_dtype is not None:
            model_kwargs['torch_dtype'] = self.autocast_dtype

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_autocast_dtype(precision: str):
        if precision == 'fp16':
            return torch.float16
        if precision == 'bf16' and hasattr(torch, 'bfloat16'):
            return torch.bfloat16
        return None

    def _prepare_inputs(self, images: List):
        inputs = self.processor(images=images, return_tensors='pt')
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    @torch.no_grad()
    def encode_pil_batch(self, images: List) -> torch.Tensor:
        if len(images) == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        inputs = self._prepare_inputs(images)
        autocast_context = nullcontext()
        if self.device.type == 'cuda' and self.autocast_dtype is not None:
            autocast_context = torch.autocast(device_type='cuda', dtype=self.autocast_dtype)

        with autocast_context:
            if hasattr(self.model, 'get_image_features'):
                embeddings = self.model.get_image_features(**inputs)
            else:
                vision_model = getattr(self.model, 'vision_model', None)
                if vision_model is None:
                    raise AttributeError('loaded model does not expose get_image_features or vision_model')
                outputs = vision_model(pixel_values=inputs['pixel_values'])
                if getattr(outputs, 'pooler_output', None) is not None:
                    embeddings = outputs.pooler_output
                elif getattr(outputs, 'last_hidden_state', None) is not None:
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    raise AttributeError('vision model output does not expose pooler_output or last_hidden_state')

        embeddings = F.normalize(embeddings.float(), dim=-1)
        return embeddings.cpu()
