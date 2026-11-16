from typing import List, Literal, Optional

import torch
from annotated_types import Ge, Le
from torch import nn
from typing_extensions import Annotated

from cerebras.modelzoo.models.multimodal.multimodal_utils import freeze_modules
from cerebras.modelzoo.models.nlp.bert.bert_pretrain_models import (
    BertClassifierHead,
)
from cerebras.modelzoo.models.vision.vision_transformer.ViTModel import (
    ViTModel,
    ViTModelConfig,
)


class ViTClassificationModelConfig(ViTModelConfig):
    name: Literal["vit_classification"]

    num_classes: int = 2
    "Number of possible classes."

    # Encoder Attn
    attention_dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.1

    # Task-specific
    use_bias_in_output: bool = True

    use_dinov2_classifier: bool = False

    freeze: Optional[List[str]] = None
    "List of regex strings used to freeze specific layers."

class PatchTokenAug(nn.Module):
    def __init__(self, grid_h, grid_w, prepend_cls_token=True):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.prepend_cls_token = prepend_cls_token

    def forward(self, tokens, aug_params, DA_layer):
        B, N, D = tokens.shape

        if self.prepend_cls_token:
            cls_tok = tokens[:, 0:1, :]
            patch_tok = tokens[:, 1:, :]
        else:
            cls_tok = None
            patch_tok = tokens

        def mask(cond):
            return cond.reshape(B, 1, 1).to(dtype=tokens.dtype)

        # 1. horizontal flip in patch-token space
        if DA_layer >= 1:
            rows = []
            for r in range(self.grid_h):
                row = patch_tok[:, r * self.grid_w:(r + 1) * self.grid_w, :]
                row_flip = torch.cat(
                    [row[:, c:c+1, :] for c in range(self.grid_w - 1, -1, -1)],
                    dim=1,
                )
                rows.append(row_flip)

            flipped = torch.cat(rows, dim=1)
            m = mask(aug_params < 0.9)
            patch_tok = patch_tok + m * (flipped - patch_tok)

        # 2. vertical flip in patch-token space
        if DA_layer >= 2:
            rows = []
            for r in range(self.grid_h - 1, -1, -1):
                row = patch_tok[:, r * self.grid_w:(r + 1) * self.grid_w, :]
                rows.append(row)

            flipped = torch.cat(rows, dim=1)
            m = mask(aug_params > 0.2)
            patch_tok = patch_tok + m * (flipped - patch_tok)

        # 3. token brightness-like scaling
        if DA_layer >= 3:
            scale = aug_params.reshape(B, 1, 1).to(dtype=tokens.dtype)
            bright = patch_tok * scale
            m = mask(aug_params > 0.2)
            patch_tok = patch_tok + m * (bright - patch_tok)

        # 4. token noise
        if DA_layer >= 4:
            noisy = patch_tok + 0.001
            m = mask(aug_params > 0.4)
            patch_tok = patch_tok + m * (noisy - patch_tok)

        # 5. patch-token crop/masking
        if DA_layer >= 5:
            keep_h = self.grid_h // 2
            keep_w = self.grid_w // 2
            top = (self.grid_h - keep_h) // 2
            left = (self.grid_w - keep_w) // 2

            rows = []
            for r in range(self.grid_h):
                row_parts = []
                for c in range(self.grid_w):
                    idx = r * self.grid_w + c
                    tok = patch_tok[:, idx:idx+1, :]

                    if top <= r < top + keep_h and left <= c < left + keep_w:
                        row_parts.append(tok)
                    else:
                        row_parts.append(tok * 0.0)

                rows.append(torch.cat(row_parts, dim=1))

            cropped = torch.cat(rows, dim=1)
            m = mask(aug_params < 0.2)
            patch_tok = patch_tok + m * (cropped - patch_tok)

        # 6. 180-degree token rotation
        if DA_layer >= 6:
            pieces = [
                patch_tok[:, i:i+1, :]
                for i in range(self.grid_h * self.grid_w - 1, -1, -1)
            ]
            rotated = torch.cat(pieces, dim=1)
            m = mask(aug_params < 0.2)
            patch_tok = patch_tok + m * (rotated - patch_tok)

        if self.prepend_cls_token:
            return torch.cat([cls_tok, patch_tok], dim=1)

        return patch_tok


class ViTClassificationModel(nn.Module):
    def __init__(self, config: ViTClassificationModelConfig):
        if isinstance(config, dict):
            config = ViTClassificationModelConfig(**config)

        super().__init__()

        self.vit_model = ViTModel(config)

        classifier_hidden_size = config.hidden_size
        self.use_dinov2_classifier = config.use_dinov2_classifier
        if self.use_dinov2_classifier:
            if not config.prepend_cls_token:
                raise ValueError(
                    f"prepend_cls_token must be set to True for Dinov2."
                )
            classifier_hidden_size *= 2

        self.classifier = BertClassifierHead(
            hidden_size=classifier_hidden_size,
            num_classes=config.num_classes,
            use_bias=config.use_bias_in_output,
            kernel_initializer=config.default_initializer,
        )

        # Freeze specified parameters
        if config.freeze is not None:
            freeze_modules(self, config.freeze)

    def reset_parameters(self):
        self.vit_model.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, input_image):
        hidden_states, pooled_states = self.vit_model(input_image)
        if self.use_dinov2_classifier:
            cls_tokens = hidden_states[:, 0]
            patch_tokens = hidden_states[:, 1:]
            pooled_states = torch.cat(
                [cls_tokens, patch_tokens.mean(dim=1)], dim=1
            )
        logits = self.classifier(pooled_states)
        return logits

    def compute_input_embeddings(self, input_image):
        input_image_embeddings = self.vit_model.embedding_layer(input_image)
        return input_image_embeddings

    def get_input_embeddings(self):
        return self.vit_model.embedding_layer

    def get_output_embeddings(self):
        """
        Extract the final layer that produces logits.
        """
        return self.classifier.classifier

    def tie_weights(self):
        # weights not tied
        pass

    def extract_features(self, input_embeddings, extract_layer_idx):
        """
        Extract features from `extract_layer_idx` of encoder
        by passing input_tensor through encoder
        input_embeddings: Tensor with output from embeddings layer
        extract_layer_idx: (inclusive)layer index in range [0, self.num_layers) (zero-indexed)
                Applies encoder layers up to (and including) `extract_layer_idx`
                instead of all encoder layers.
                For ex: extract_layer_idx=3 would run fwd pass from encoder_block_0 to encoder_block_3
                and return outputs from encoder_block_3.
                If `extract_layer_idx` = None and `norm` != None, then
                the output returned would be encoder_block_{self.num_layers-1} -> norm -> output (return).
        """

        return self.vit_model.extract_features(
            input_embeddings, extract_layer_idx
        )

class ViTClassificationModelWithTokenAug(ViTClassificationModel):
    def __init__(self, config):
        super().__init__(config)

        image_h, image_w = config.image_size
        patch_h, patch_w = config.patch_size

        self.grid_h = image_h // patch_h
        self.grid_w = image_w // patch_w
        self.prepend_cls_token = config.prepend_cls_token

        self.token_aug = PatchTokenAug(
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            prepend_cls_token=self.prepend_cls_token,
        )

    def forward(self, input_image, aug_params=None, DA_layer=0):
        input_embeddings = self.vit_model.embedding_layer(input_image)
        print("TOKEN AUG RUNNING")
        if self.training and aug_params is not None:
            input_embeddings = self.token_aug(
                input_embeddings,
                aug_params,
                DA_layer,
            )

        hidden_states = self.vit_model.extract_features(
            input_embeddings,
            extract_layer_idx=None,
        )

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        if self.use_dinov2_classifier:
            cls_tokens = hidden_states[:, 0]
            patch_tokens = hidden_states[:, 1:]
            pooled_states = torch.cat(
                [cls_tokens, patch_tokens.mean(dim=1)],
                dim=1,
            )
        else:
            if self.prepend_cls_token:
                pooled_states = hidden_states[:, 0]
            else:
                pooled_states = hidden_states.mean(dim=1)

        logits = self.classifier(pooled_states)
        return logits