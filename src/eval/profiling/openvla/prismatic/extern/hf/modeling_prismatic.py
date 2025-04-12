"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch

from torch import nn, pi
import torch.nn.functional as F
from einops import rearrange, repeat
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# Get Logger
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            # Dispatch to Language Model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats
        self.default_action_decoding_strategy = config.default_action_decoding_strategy

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)  # 256  [-1, -1 + 2/256, -1 + 4/256, ..., 1]
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0  # 255  [-1 + 1/256, -1 + 3/256, ..., 1 - 1/256]

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, 
        return_logits: bool = False, **kwargs: str
    ) -> dict[str, np.ndarray]:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them.

        Args:
            input_ids: Input token IDs
            unnorm_key: Key for unnormalization statistics
            return_logits: Whether to return logits and probabilities
            **kwargs: Additional arguments for model forward pass
            
        Returns:
            Dictionary containing:
                - 'actions': Unnormalized actions (numpy array)
                - If return_logits=True, also includes:
                    - 'action_probs': Full probability distributions for all tokens for all action dimensions
        """
        # Ensure input has the special token
        input_ids = self._ensure_special_token(input_ids)

        action_decoding_strategy = self.get_action_decoding_strategy(unnorm_key)
        action_dim = self.get_action_dim(unnorm_key, action_decoding_strategy)

        # Run model inference
        if return_logits:
            # Add the necessary parameters to get scores during generation
            generate_kwargs = {
                **kwargs,
                "return_dict_in_generate": True,
                "output_scores": True,
                "max_new_tokens": action_dim
            }
            
            # Run inference with logits
            generation_output = self.generate(input_ids, **generate_kwargs)
            generated_ids = generation_output.sequences
            all_token_scores = generation_output.scores
            
            debug_action_tokens = torch.tensor([[torch.argmax(all_token_scores[0][-1], dim=-1)]], device=generated_ids.device)
            logger.debug(f"debug_action_tokens: {debug_action_tokens}")

            probs_over_bin_centers_list = self._get_action_probs_from_logits(all_token_scores)
            logger.debug(f"probs_over_bin_centers: {probs_over_bin_centers_list}")

            # Unnormalize action probabilities to dataset action space
            unnormalized_probs_by_dimension = self._unnormalize_action_probs(probs_over_bin_centers_list, unnorm_key)
            logger.debug(f"unnormalized_action_probs: {unnormalized_probs_by_dimension}")
        else:
            generated_ids = self.generate(input_ids, max_new_tokens=action_dim, **kwargs)

        # Extract predicted action tokens and translate into (normalized) continuous actions (OpenVLA standard)
        openvla_normalized_actions = self._tokens_to_normalized_actions(
            generated_ids, action_dim
        )

        debug_normalized_actions = self._tokens_to_normalized_actions(debug_action_tokens, action_dim)
        logger.debug(f"debug_normalized_actions: {debug_normalized_actions}")

        # Unnormalize and discretize (if needed) actions
        actions = self._unnormalize_actions(
            normalized_actions=openvla_normalized_actions,
            unnorm_key=unnorm_key,
            action_decoding_strategy=action_decoding_strategy
        )

        if actions[0] != np.argmax(unnormalized_probs_by_dimension[0]):
            logger.warning("MISMATCH")
            probs_over_bin_centers_list = self._get_action_probs_from_logits(all_token_scores)
            logger.debug(f"probs_over_bin_centers: {probs_over_bin_centers_list}")

            # Unnormalize action probabilities to dataset action space
            unnormalized_probs_by_dimension = self._unnormalize_action_probs(probs_over_bin_centers_list, unnorm_key, int(actions[0]), int(np.argmax(unnormalized_probs_by_dimension[0])), should_break=True)
            logger.debug(f"unnormalized_action_probs: {unnormalized_probs_by_dimension}")


        debug_actions = self._unnormalize_actions(
            normalized_actions=debug_normalized_actions,
            unnorm_key=unnorm_key,
            action_decoding_strategy=action_decoding_strategy
        )
        logger.debug(f"debug_actions: {debug_actions}")

        result = {'actions': actions, 'debug_actions': debug_actions}

        if return_logits:
            result.update({
                'action_probs_by_dimension': unnormalized_probs_by_dimension
            })

        return result
    
    def _ensure_special_token(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Ensure input_ids has the special empty token."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        return input_ids

    def _tokens_to_normalized_actions(self, generated_ids: torch.Tensor, action_dim: int) -> np.ndarray:
        logger.debug("\n=== _tokens_to_normalized_actions DEBUG ===")
        logger.debug(f"Input generated_ids shape: {generated_ids.shape}")
        
        predicted_tokens = generated_ids[0, -action_dim:].cpu().numpy()
        logger.debug(f"Predicted tokens: {predicted_tokens}")
        
        # Step 1: Convert tokens to bin indices   Llama 32000 -> 255 action tokens
        # 31999 -> 
        discretized_actions = np.clip(self.vocab_size - predicted_tokens - 1,
                                    a_min=0, a_max=self.bin_centers.shape[0] - 1)
        logger.debug(f"Token to bin mapping:")
        
        # Track token ranges and their mappings
        token_ranges = {
            "non_action": [],
            "action": [],
            "padding": []
        }
        
        for token, bin_idx in zip(predicted_tokens, discretized_actions):
            logger.debug(f"Token {token} → Bin {bin_idx}")
            if token < self.vocab_size - self.config.n_action_bins:
                token_ranges["non_action"].append(token)
            elif token >= self.vocab_size:
                token_ranges["padding"].append(token)
            else:
                token_ranges["action"].append(token)
        
        # Log token range statistics
        logger.debug("\nToken Range Statistics:")
        for range_name, tokens in token_ranges.items():
            logger.debug(f"{range_name} tokens: {len(tokens)} tokens, values: {tokens}")
        
        # Step 2: Get normalized values
        normalized_actions = self.bin_centers[discretized_actions]
        logger.debug(f"\nNormalized actions: {normalized_actions}")
        logger.debug(f"Bin centers shape: {self.bin_centers.shape}")
        logger.debug(f"Min/Max bin centers: {self.bin_centers.min():.4f}/{self.bin_centers.max():.4f}")
        logger.debug(f"Min/Max normalized actions: {normalized_actions.min():.4f}/{normalized_actions.max():.4f}")
        
        return normalized_actions
    
    def _get_action_probs_from_logits(self, logits: torch.Tensor) -> list[np.ndarray]:
        """Get the full probability distribution for all action tokens for each action dimension.

        This function maps logits to action probabilities following OpenVLA's action token mapping:

        1. The vocabulary consists of:
           - Non-action tokens: indices [0, 31743]
           - Action tokens: indices [31744, 31999] (256 tokens)
           - Padding tokens: indices [32000, 32063]

        2. Action token mapping:
           - All non-action tokens (< 31744) map to first bin center (0)
           - Action tokens map sequentially to bin centers 0-253
           - Last two action tokens (254,255) map to last bin center (254)
           - Any padding tokens map to last bin center (254)

        3. Final probabilities are reversed to match action decoding logic:
           self.vocab_size - predicted_tokens - 1

        Args:
            logits: Logits tensor for each action dimension [batch_size, n_dims, vocab_size]

        Returns:
            List of probability arrays, one per action dimension
        """
        n_action_tokens = self.config.n_action_bins  # 256 tokens
        n_bin_centers = self.bin_centers.shape[0]    # 255 centers
        first_action_idx = self.vocab_size - n_action_tokens  # 32000 - 256 = 31744 -> llama idx corresponding to last bin
        
        # DEBUG: Log input shapes and key indices
        logger.debug("=== _get_action_probs_from_logits DEBUG ===")
        logger.debug(f"Input logits shape: {[l.shape for l in logits]}")
        logger.debug(f"n_action_tokens: {n_action_tokens}")
        logger.debug(f"n_bin_centers: {n_bin_centers}")
        logger.debug(f"first_action_idx: {first_action_idx}")
        logger.debug(f"vocab_size: {self.vocab_size}")
        
        action_probs_list = []
        
        for dim_idx, dim_logits in enumerate(logits[0]):
            logger.debug(f"\n=== Processing Dimension {dim_idx} ===")
            
            # Step 1: Create bin probability tensor [255]
            bin_probs = torch.zeros(n_bin_centers, device=dim_logits.device, dtype=dim_logits.dtype)
            
            # Step 2: Convert logits to probabilities
            vocab_probs = torch.softmax(dim_logits, dim=-1)
            logger.debug(f"Sum of vocab_probs: {vocab_probs.sum():.4f}")
            
            # Step 3: Map non-action tokens
            non_action_prob = torch.sum(vocab_probs[:first_action_idx + 1])  # [:31745]
            bin_probs[-1] = non_action_prob
            # bin_probs[0] = non_action_prob

            logger.debug(f"Non-action tokens (mapped to last bin) probability: {non_action_prob:.4f}")
            
            # Step 4: Map action tokens with reverse mapping
            logger.debug("\nSignificant action token mappings:")
            for offset in range(1, n_bin_centers): # 1 -> 254
                token_idx = first_action_idx + offset  # 31744 + 1 = 31745, 31744 + 254 = 31998
                bin_idx = n_bin_centers - offset  # 255 - 254 = 1, 255 - 1 = 254  # Reverse mapping
                # bin_idx = offset
                prob = vocab_probs[token_idx]
                bin_probs[bin_idx] += prob  # bin_probs[1] to bin_probs[254]
                if prob > 0.01:
                    logger.debug(f"Token {token_idx} → Bin {bin_idx}: {prob:.4f}")
                    logger.debug(f"  Corresponding normalized value: {self.bin_centers[bin_idx]:.4f}")
            
            # Step 5: Map padding tokens
            padding_prob = torch.sum(vocab_probs[first_action_idx + n_bin_centers:])  # [31744 + 255:] = [31999:]
            bin_probs[0] += padding_prob
            # bin_probs[-1] += padding_prob

            logger.debug(f"Padding tokens (mapped to first bin) probability: {padding_prob:.4f}")
            
            # Convert to numpy and verify
            action_probs = bin_probs.detach().cpu().numpy()
            logger.debug(f"\nFinal bin probabilities sum: {action_probs.sum():.4f}")
            argmax_bin = np.argmax(action_probs)
            logger.debug(f"Argmax bin: {argmax_bin}")
            logger.debug(f"Corresponding normalized value: {self.bin_centers[argmax_bin]:.4f}")
            
            # For values near integer boundaries
            if abs(self.bin_centers[argmax_bin] - round(self.bin_centers[argmax_bin])) < 0.1:
                logger.debug(f"Near-boundary bin {argmax_bin}: value={self.bin_centers[argmax_bin]:.4f} → action={argmax_bin}")
            
            action_probs_list.append(action_probs)
        
        return action_probs_list

    def _unnormalize_actions(
        self, normalized_actions: np.ndarray, unnorm_key: Optional[str], action_decoding_strategy: str
    ) -> np.ndarray:
        """Unnormalize actions based on the decoding strategy."""
        if action_decoding_strategy in ["naive_dim_extension", "simple_mapping"]:
            # Get action statistics
            action_stats = self.get_action_stats(unnorm_key)
            action_high = np.array(action_stats["q99"])
            action_low = np.array(action_stats["q01"])
            mask = action_stats.get("mask", np.ones_like(action_low, dtype=bool))
            discrete = np.array(action_stats.get("discrete", np.zeros_like(mask, dtype=bool)))
            
            # Scale from [-1,1] to [0,1] to [low,high]
            actions = 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low
            
            # Round discrete dimensions
            if np.any(discrete):
                actions[discrete] = np.round(actions[discrete], decimals=0)
            
            # Apply mask
            actions = np.where(mask, actions, normalized_actions)
        elif action_decoding_strategy == "manual_rule_mapping":
            actions = normalized_actions
        else:
            raise ValueError(f"Unknown action decoding strategy: {action_decoding_strategy} for {unnorm_key}")
        
        return actions

    def _unnormalize_action_probs(
        self, action_probs_by_dimension: list[np.ndarray], unnorm_key: Optional[str], pred: int = 0, argmax_act: int = 0, should_break: bool = False
    ) -> list[np.ndarray]:
        logger.debug("\n=== _unnormalize_action_probs DEBUG ===")
        
        # Get action statistics
        action_stats = self.get_action_stats(unnorm_key)
        action_upper_bound = np.array(action_stats["q99"])
        action_lower_bound = np.array(action_stats["q01"])
        logger.debug(f"Action bounds: [{action_lower_bound}, {action_upper_bound}]")
        
        unnormalized_probs_by_dimension = []
        
        for dim_idx, dimension_probs in enumerate(action_probs_by_dimension):
            logger.debug(f"\n=== Processing Dimension {dim_idx} ===")
            dimension_upper = action_upper_bound[dim_idx]
            dimension_lower = action_lower_bound[dim_idx]
            
            # Step 1: Unnormalize bin centers
            unnormalized_bin_centers = 0.5 * (self.bin_centers + 1) * (dimension_upper - dimension_lower) + dimension_lower
            logger.debug(f"First few unnormalized bin centers: {unnormalized_bin_centers[:5]}")
            logger.debug(f"Last few unnormalized bin centers: {unnormalized_bin_centers[-5:]}")
            
            # Log probability distribution statistics before mapping
            logger.debug("\nProbability Distribution Statistics (Before Mapping):")
            logger.debug(f"Total probability sum: {np.sum(dimension_probs):.4f}")
            logger.debug(f"Number of non-zero probabilities: {np.count_nonzero(dimension_probs)}")
            logger.debug(f"Max probability: {np.max(dimension_probs):.4f} at bin {np.argmax(dimension_probs)}")
            
            # Group probabilities by ranges
            low_probs = dimension_probs[dimension_probs < 0.01]
            mid_probs = dimension_probs[(dimension_probs >= 0.01) & (dimension_probs < 0.1)]
            high_probs = dimension_probs[dimension_probs >= 0.1]

            logger.debug("\nProbability Range Analysis:")
            logger.debug(f"Low probs (<0.01): count={len(low_probs)}, sum={np.sum(low_probs):.4f}")
            logger.debug(f"Mid probs (0.01-0.1): count={len(mid_probs)}, sum={np.sum(mid_probs):.4f}")
            logger.debug(f"High probs (>0.1): count={len(high_probs)}, sum={np.sum(high_probs):.4f}")

            # Step 2: Calculate discrete action range
            min_discrete_action = int(np.floor(dimension_lower)) # q01 = 0.4 -> 0
            max_discrete_action = int(np.ceil(dimension_upper)) # q99 = 7.8 -> 8
            discrete_action_range = max_discrete_action - min_discrete_action + 1 # 9
            logger.debug(f"\nDiscrete action range: [{min_discrete_action}, {max_discrete_action}] ({discrete_action_range} values)")

            # Step 3: Initialize and map probabilities
            dimension_unnormalized_probs = np.zeros(discrete_action_range)  # [0] * 9 

            # Log the top probability bins before mapping
            top_k = 5
            top_indices = np.argsort(dimension_probs)[-top_k:][::-1]
            logger.debug("\nTop input probabilities before mapping:")
            for idx, bin_idx in enumerate(top_indices):
                logger.debug(f"Bin {bin_idx}: {dimension_probs[bin_idx]:.4f} → Value {self.bin_centers[bin_idx]:.4f}")

            # Step 4: Map and accumulate probabilities
            mapped_prob_sum = 0.0
            unmapped_prob_sum = 0.0
            action_mappings = {}
            
            for bin_idx, (bin_center, probability) in enumerate(zip(unnormalized_bin_centers, dimension_probs)):
                discrete_action = int(np.round(bin_center, 0))
                array_index = discrete_action - min_discrete_action  # shift to 0-indexed
                
                # Ensure index is within bounds
                if 0 <= array_index < discrete_action_range:  # 0 -> 8
                    dimension_unnormalized_probs[array_index] += probability
                    mapped_prob_sum += probability
                    
                    # Track mappings for logging
                    if discrete_action not in action_mappings:
                        action_mappings[discrete_action] = {"centers": [], "probs": []}
                    action_mappings[discrete_action]["centers"].append(bin_center)
                    action_mappings[discrete_action]["probs"].append(probability)
                    
                    if probability > 0.01:
                        logger.debug(f"Mapping bin {bin_idx} (value={bin_center:.4f}, prob={probability:.4f}) → action {discrete_action}")
                else:
                    unmapped_prob_sum += probability
                    if probability > 0.01:
                        logger.debug(f"WARNING: Unmapped significant probability {probability:.4f} at bin {bin_idx} (value={bin_center:.4f})")
            
            if should_break:
                top2_indices = np.argsort(dimension_unnormalized_probs)[-2:][::-1]
                debug_top_2_probs = []
                debug_top_2_probs.append({
                    "bin_idx": top2_indices.tolist(),
                    "prob": dimension_unnormalized_probs[top2_indices].tolist(),
                    "pred": pred,
                    "argmax_act": argmax_act
                })
                logger.debug(f"Top 2 probabilities:")
                logger.debug(f"  1st: bin {top2_indices[0]} -> prob {dimension_unnormalized_probs[top2_indices[0]]:.4f}")
                logger.debug(f"  2nd: bin {top2_indices[1]} -> prob {dimension_unnormalized_probs[top2_indices[1]]:.4f}")

                import os
                import json
                if not os.path.exists("debug_top_2_probs.json"):
                    with open("debug_top_2_probs.json", "w") as f:
                        json.dump(debug_top_2_probs, f)
                else:
                    with open("debug_top_2_probs.json", "r") as f:
                        existing_mappings = json.load(f)
                    existing_mappings.append(debug_top_2_probs)
                    with open("debug_top_2_probs.json", "w") as f:
                        json.dump(existing_mappings, f)

            logger.debug(f"\nProbability Mapping Statistics:")
            logger.debug(f"Total mapped probability: {mapped_prob_sum:.4f}")
            logger.debug(f"Total unmapped probability: {unmapped_prob_sum:.4f}")
            # Log accumulated mappings
            logger.debug("\nBin center and probability mappings by discrete action:")
            for action in sorted(action_mappings.keys()):
                centers = action_mappings[action]["centers"]
                probs = action_mappings[action]["probs"]
                logger.debug(f"discrete action {action}: bin center values {', '.join([f'{x:.6f}' for x in centers])}")
                logger.debug(f"discrete action {action}: probs {' + '.join([f'{x:.6f}' for x in probs])} = {sum(probs):.6f}")
            
            # Step 5: Normalize and verify
            original_sum = np.sum(dimension_unnormalized_probs)
            dimension_unnormalized_probs = dimension_unnormalized_probs / np.sum(dimension_unnormalized_probs)

            logger.debug(f"\nNormalization Statistics:")
            logger.debug(f"Sum before normalization: {original_sum:.4f}")
            logger.debug(f"Sum after normalization: {np.sum(dimension_unnormalized_probs):.4f}")

            argmax_action = np.argmax(dimension_unnormalized_probs) + min_discrete_action
            logger.debug(f"Argmax unnormalized action: {argmax_action}")
            logger.debug(f"Corresponding probability: {dimension_unnormalized_probs[argmax_action - min_discrete_action]:.4f}")

            # Log final probability distribution statistics
            logger.debug("\nFinal Probability Distribution Statistics:")
            logger.debug(f"Number of non-zero probabilities: {np.count_nonzero(dimension_unnormalized_probs)}")
            logger.debug(f"Min/Max probabilities: {np.min(dimension_unnormalized_probs[dimension_unnormalized_probs > 0]):.4f}/{np.max(dimension_unnormalized_probs):.4f}")
            
            # For values near integer boundaries
            if abs(self.bin_centers[argmax_action] - round(self.bin_centers[argmax_action])) < 0.1:
                logger.debug(f"Near-boundary bin {argmax_action}: value={self.bin_centers[argmax_action]:.4f} → action={argmax_action}")
            
            unnormalized_probs_by_dimension.append(dimension_unnormalized_probs)
        
        return unnormalized_probs_by_dimension

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {', '.join(norm_stats.keys())}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        if unnorm_key not in norm_stats:
            raise ValueError(
                f"The unnorm_key {unnorm_key} is missing in dataset statistics, "
                f"Existing dataset statistics: {', '.join(norm_stats.keys())}"
            )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None, decoding_strategy: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        decoding_strategy = decoding_strategy or self.get_action_decoding_strategy(unnorm_key)

        if decoding_strategy == "manual_rule_mapping":  # use OpenVLA standard ACTION_DIM=7
            return 7 # OpenVLA standard action dimension
        elif decoding_strategy == "naive_dim_extension" or decoding_strategy == "simple_mapping":  # use dataset-specific action dimension
            return len(self.norm_stats[unnorm_key]["action"]["q01"])
        else:
            raise ValueError(f"Unknown action decoding strategy: {decoding_strategy}")

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def get_action_decoding_strategy(self, unnorm_key: Optional[str] = None) -> str:
        """Get the decoding strategy used for actions."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key].get("action_decoding_strategy", self.default_action_decoding_strategy)
