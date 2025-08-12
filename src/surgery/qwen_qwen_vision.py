import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig,
    PreTrainedModel, PretrainedConfig, AutoConfig
)
from typing import Optional, Tuple
from pathlib import Path
import json
from loguru import logger
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


#   1. The Qwen2.5-VL visual model has a merger component that contains:
#     - merger.mlp.0: Linear(5120 → 5120)
#     - merger.mlp.1: GELU()
#     - merger.mlp.2: Linear(5120 → 3584)

# === Define Configuration Class ===
class QwenQwenHybridConfig(PretrainedConfig):
    model_type = "qwen_qwen_hybrid"  # Unique identifier for AutoModel registration

    def __init__(
        self,
        vision_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        language_model_name="Qwen/Qwen3-4B",
        disable_pooling=True,
        # These will be populated during model initialization
        vision_placeholder_id=None,
        vision_start_token_id=None,
        vision_end_token_id=None,
        **kwargs,
    ):
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.disable_pooling = disable_pooling
        self.vision_placeholder_id = vision_placeholder_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        
        # Ensure architectures is set for HF compatibility
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["QwenQwenHybrid"]
            
        super().__init__(**kwargs)


# Inherit from PreTrainedModel
class QwenQwenHybrid(PreTrainedModel):
    """
    Qwen2.5-VL vision + Qwen3-4B language model hybrid.
    - Same model family for better compatibility
    - Simpler adapter without complex stabilization
    - Native thinking support in Qwen3
    """
    config_class = QwenQwenHybridConfig

    # Update __init__ to be config-based
    def __init__(self, config: QwenQwenHybridConfig):
        # Initialize PreTrainedModel
        super().__init__(config)

        # Add vision cache for inference
        self._vision_cache = None
        self._cached_input_ids = None
        
        # Store pooling preference from config
        self.disable_pooling = config.disable_pooling
        
        # Extract names from config
        vision_model_name = config.vision_model_name
        language_model_name = config.language_model_name

        # --- 1. Load Original Models ---
        logger.info(f"Loading Qwen2.5-VL vision components from: {vision_model_name}")
        # Import the specific Qwen2.5-VL model class
        from transformers import Qwen2_5_VLForConditionalGeneration
        logger.debug(f"About to load Qwen2_5_VLForConditionalGeneration from {vision_model_name}")
        # Load without device_map first to avoid meta tensors
        # DDP support: Place vision model on same device as language model
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank >= 0:
            # DDP mode: load on CPU first, then move to specific GPU
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vision_model_name,
                torch_dtype=torch.float16,
                device_map={"": local_rank},
            )
        else:
            # Single GPU mode: simple load
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vision_model_name,
                torch_dtype=torch.float16,
            )
        logger.info(f"Successfully loaded model of type: {type(qwen_model)}")
        # Access visual model directly from the model
        self.vision_model = qwen_model.visual
        # In Qwen2.5-VL, the vision model has the projection built-in
        self.vision_hidden_size = qwen_model.config.vision_config.hidden_size
        qwen_output_dim = qwen_model.config.vision_config.out_hidden_size
        self.spatial_merge_size = qwen_model.config.vision_config.spatial_merge_size

        logger.info(f"Loading Qwen3-4B from: {language_model_name}")

        # Configure 8-bit quantization for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,  # Use bf16 to avoid casts
            # For 8-bit, we don't use nf4 (that's for 4-bit)
        )

        # Robust Flash Attention enablement
        from transformers.utils import is_flash_attn_2_available
        
        flash_ok = False
        try:
            flash_ok = is_flash_attn_2_available()
        except Exception:
            pass
        
        extra_load_kwargs = {}
        if flash_ok:
            # Newer HF: pass at load
            extra_load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 available, enabling...")
        else:
            logger.info("Flash Attention 2 not available, using SDPA (still efficient)")
        
        # Load model in 8-bit for memory efficiency (but no LoRA)
        # DDP support: Use local rank for device_map if DDP is active
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank >= 0:
            # DDP mode: place model on specific GPU
            device_map = {"": local_rank}
            logger.info(f"DDP mode: loading model on device {local_rank}")
        else:
            # Single GPU mode: use auto
            device_map = "auto"
            logger.info("Single GPU mode: using device_map='auto'")
        
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name,
            quantization_config=bnb_config,
            device_map=device_map,  # DDP-aware device placement
            torch_dtype=torch.bfloat16,  # Use bf16 to avoid casts
            **extra_load_kwargs,
        )
        
        # Post-load: set whichever attribute exists
        if flash_ok:
            for obj in (self.language_model.config, getattr(self.language_model, "model", None)):
                if obj is None:
                    continue
                if hasattr(obj, "attn_implementation"):
                    setattr(obj, "attn_implementation", "flash_attention_2")
                elif hasattr(obj, "_attn_implementation"):
                    setattr(obj, "_attn_implementation", "flash_attention_2")
        
        # Enable PyTorch SDPA kernels as fallback
        import torch.backends.cuda as cuda
        cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
        
        # Safe logging (won't crash if attribute name differs)
        attn_impl = (
            getattr(self.language_model.config, "attn_implementation", None)
            or getattr(self.language_model.config, "_attn_implementation", None)
            or "sdpa/auto"
        )
        logger.info(f"Attention implementation: {attn_impl}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.r1_hidden_dim = self.language_model.config.hidden_size

        # Freeze language model - we only train the merger MLP
        for param in self.language_model.parameters():
            param.requires_grad = False

        logger.info("Language model loaded in 8-bit and frozen - only training merger MLP layers")

        # --- 2. Replace Qwen's merger projection to match Qwen3's dimension ---
        logger.info(f"Replacing Qwen merger projection: {qwen_output_dim} -> {self.r1_hidden_dim}")

        # Store original layer 2 for backup
        original_layer2 = self.vision_model.merger.mlp[2]

        # Create wrapper that handles dtype conversion
        class DtypeWrapper(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                # Add pre-LayerNorm for stability (fp32 by default)
                self.pre_ln = nn.LayerNorm(in_dim, eps=1e-5)
                # Use fp32 for stability
                self.linear = nn.Linear(in_dim, out_dim, bias=True, dtype=torch.float32)
                # ZERO-INIT for safety - adapter starts "off"
                nn.init.zeros_(self.linear.weight)
                nn.init.zeros_(self.linear.bias)

            def forward(self, x):
                # Do LayerNorm and Linear in fp32 for numerical stability
                orig_dtype = x.dtype
                x = x.to(torch.float32)  # Convert to fp32 for stable LayerNorm
                x = self.pre_ln(x)  # Pre-normalize in fp32
                # Guard against bad upstream values
                x = torch.clamp(x, -10.0, 10.0)
                x = self.linear(x)  # Linear also in fp32
                x = x.to(orig_dtype)  # Convert back to original dtype (bf16)
                return x

        # Replace with our wrapper
        self.vision_model.merger.mlp[2] = DtypeWrapper(5120, self.r1_hidden_dim)

        # CRITICAL: Manually set requires_grad=True for the replaced layer
        self.vision_model.merger.mlp[2].linear.weight.requires_grad = True
        self.vision_model.merger.mlp[2].linear.bias.requires_grad = True

        # No gradient hooks needed - AdaFactor handles stability

        # Get the actual vision token IDs from the tokenizer
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        logger.info("Vision tokens found in Qwen3 tokenizer:")
        logger.info(f"  <|vision_start|>: {vision_start_id}")
        logger.info(f"  <|vision_end|>: {vision_end_id}")
        logger.info(f"  <|image_pad|>: {image_pad_id}")

        # Update the config object (self.config)
        self.config.vision_start_token_id = vision_start_id
        self.config.vision_end_token_id = vision_end_id
        self.config.vision_placeholder_id = image_pad_id
        
        # Synchronize crucial attributes from the LM config to the main config
        self.config.eos_token_id = self.language_model.config.eos_token_id
        # Handle potential None pad_token_id
        self.config.pad_token_id = getattr(self.language_model.config, 'pad_token_id', self.tokenizer.pad_token_id)
        self.config.bos_token_id = self.language_model.config.bos_token_id
        self.config.vocab_size = self.language_model.config.vocab_size

        base_embeds = self.language_model.get_input_embeddings()
        # Register as parameters - ensure they require grad
        self.vision_start_embedding = nn.Parameter(base_embeds.weight[self.config.vision_start_token_id].clone())
        self.vision_end_embedding = nn.Parameter(base_embeds.weight[self.config.vision_end_token_id].clone())
        self.vision_start_embedding.requires_grad = True
        self.vision_end_embedding.requires_grad = True

        # Add post-projection LayerNorm for stability (create BEFORE freezing)
        # Use bfloat16 to match vision embeddings dtype
        self.post_proj_ln = nn.LayerNorm(self.r1_hidden_dim, dtype=torch.bfloat16)  # Post-norm on LLM dims

        # Always create extra_gate for checkpoint compatibility
        self.extra_gate = nn.Parameter(torch.zeros(1))  # Learnable gate, tanh() ~ 0 at init
        
        # Curriculum knobs for token resampling (only used when pooling is enabled)
        if self.disable_pooling:
            logger.info("Vision pooling DISABLED: all vision tokens will be passed through")
        else:
            self.base_k = 16  # Start with small number of pooled tokens
            self.max_vision_tokens = 128  # Start conservative, will be ramped by curriculum
            self.allow_extra_tokens = False  # Will be enabled by curriculum
            self.use_soft_gate = True
            logger.info(f"Vision pooling ENABLED: max_tokens={self.max_vision_tokens}, base_k={self.base_k}")

        del qwen_model
        self._freeze_base_models()

        # CRITICAL: Put vision model in eval mode to prevent BatchNorm updates
        # This prevents the model from updating running statistics during training
        self.vision_model.eval()

        # Don't set the device attribute - PreTrainedModel handles this

        # Log trainable parameters info
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

        # List what's trainable
        logger.info("Trainable components:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                logger.debug(f"  {name}: {param.shape}")

    # === Helper method for Training Script Compatibility ===
    @classmethod
    def from_base_models(cls,
                         vision_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                         language_model_name="Qwen/Qwen3-4B",
                         disable_pooling=True):
        """Helper method to initialize the model from base components, used by the training script."""
        logger.info("Initializing QwenQwenHybrid using from_base_models (Training mode).")
        # Create the configuration first
        config = QwenQwenHybridConfig(
            vision_model_name=vision_model_name,
            language_model_name=language_model_name,
            disable_pooling=disable_pooling
        )
        # Initialize the model using the config (calls __init__)
        model = cls(config)
        return model

    def get_input_embeddings(self):
        """Get input embeddings from the language model."""
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings for the language model."""
        self.language_model.set_input_embeddings(value)
    
    # === NEW: Required Generation Methods ===
    def get_output_embeddings(self):
        """Delegate to the language model's output embeddings (the 'lm_head')."""
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """
        Prepares inputs for the next step of generation.
        Distinguishes between prefill (with vision) and decoding (last token only).
        """
        
        if past_key_values is not None:
            # Decoding phase (cache is present)
            # 1. Only pass the last token generated
            input_ids = input_ids[:, -1:]
            # 2. Vision inputs are no longer needed (context is in the cache)
            pixel_values = None
            image_grid_thw = None
        else:
            # Prefill phase (first step)
            # Keep vision inputs if provided in kwargs
            pixel_values = kwargs.get("pixel_values", None)
            image_grid_thw = kwargs.get("image_grid_thw", None)

        # Construct the dictionary passed to model.forward()
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "use_cache": kwargs.get("use_cache", True),
        }
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Required for beam search support."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the language model."""
        # Only enable for language model - vision model is frozen
        if hasattr(self.language_model, 'gradient_checkpointing_enable'):
            self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        # DO NOT enable gradient checkpointing for frozen vision model
        # It causes NaN gradients when the model is frozen

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the language model."""
        if hasattr(self.language_model, 'gradient_checkpointing_disable'):
            self.language_model.gradient_checkpointing_disable()
        if hasattr(self.vision_model, 'gradient_checkpointing_disable'):
            self.vision_model.gradient_checkpointing_disable()

    def train(self, mode=True):
        """Override train to keep vision model in eval mode."""
        super().train(mode)
        # ALWAYS keep vision model in eval mode to prevent BatchNorm updates
        self.vision_model.eval()
        return self

    def _freeze_base_models(self):
        """Freeze all components except the entire merger MLP and markers."""
        # Freeze ALL vision model parameters first
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Then unfreeze the ENTIRE merger MLP (all 3 layers)
        # Layer 0: Linear(5120 -> 5120)
        self.vision_model.merger.mlp[0].weight.requires_grad = True
        self.vision_model.merger.mlp[0].bias.requires_grad = True
        # Layer 1 is GELU, no parameters
        # Layer 2: Our replaced DtypeWrapper(5120 -> 2560)
        # Make sure both pre_ln and linear are trainable
        for param in self.vision_model.merger.mlp[2].parameters():
            param.requires_grad = True

        # Freeze language model completely
        for param in self.language_model.parameters():
            param.requires_grad = False

        # Ensure markers are trainable
        self.vision_start_embedding.requires_grad = True
        self.vision_end_embedding.requires_grad = True

    def clear_vision_cache(self):
        """Clear the vision cache for new images."""
        self._vision_cache = None
        self._cached_input_ids = None
        logger.debug("Cleared vision cache")

    def _resample_tokens(self, vis_tokens: torch.Tensor, k: int) -> torch.Tensor:
        """
        vis_tokens: [T, D] (tokens for ONE image after merger projection)
        Return k pooled tokens by fixed-window average.
        """
        T, D = vis_tokens.shape
        if k >= T:
            return vis_tokens  # nothing to pool

        # Split tokens into k nearly-equal windows and mean-pool each window
        idx = torch.linspace(0, T, steps=k+1, device=vis_tokens.device).long()
        out = []
        for i in range(k):
            s, e = idx[i].item(), idx[i+1].item()
            # safeguard for empty windows
            if e <= s:
                e = min(s+1, T)
            out.append(vis_tokens[s:e].mean(dim=0, keepdim=True))
        return torch.cat(out, dim=0)  # [k, D]

    def _global_token(self, vis_tokens: torch.Tensor) -> torch.Tensor:
        # Robust "CLS" surrogate: mean-pool all visual tokens
        return vis_tokens.mean(dim=0, keepdim=True)  # [1, D]

    def get_trainable_parameters(self):
        """Return all trainable parameters (entire merger MLP + markers + post_proj_ln + extra_gate)."""
        trainable_params = []

        # Get ALL merger MLP parameters
        # Layer 0
        trainable_params.extend([self.vision_model.merger.mlp[0].weight, self.vision_model.merger.mlp[0].bias])
        # Layer 2 (our replaced layer) - includes pre_ln parameters
        for param in self.vision_model.merger.mlp[2].parameters():
            trainable_params.append(param)

        # Add marker embeddings
        trainable_params.extend([self.vision_start_embedding, self.vision_end_embedding])
        
        # Add post-projection LayerNorm parameters
        trainable_params.extend([self.post_proj_ln.weight, self.post_proj_ln.bias])
        
        # Add curriculum gate
        trainable_params.append(self.extra_gate)

        return trainable_params

    def get_vision_embeddings(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Process images through Qwen's vision pipeline."""
        # Quick and dirty cache for inference
        if self._vision_cache is not None and not self.training:
            logger.debug("Using cached vision embeddings")
            return self._vision_cache

        # Debug input
        if not hasattr(self, '_vision_debug_count'):
            self._vision_debug_count = 0

        if self._vision_debug_count < 3:
            logger.debug(f"Vision Input {self._vision_debug_count}:")
            logger.debug(f"  pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}")
            logger.debug(f"  pixel min/max: {pixel_values.min().item():.4f} / {pixel_values.max().item():.4f}")
            logger.debug(f"  has NaN: {torch.isnan(pixel_values).any().item()}")
            self._vision_debug_count += 1

        # Convert to float32 for numerical stability
        pixel_values = pixel_values.to(dtype=torch.float32)

        # Process vision model WITHOUT torch.no_grad()!
        # The model is frozen via requires_grad=False but we MUST allow gradient flow
        # through it to reach our trainable merger projection
        # This is CRITICAL - no_grad breaks training completely!

        # Disable autocast for vision path to ensure numerical stability
        # The DtypeWrapper already handles fp32 conversion where needed
        with torch.amp.autocast('cuda', enabled=False):
            # Use the vision model's forward method
            # Skip pre-norm due to Qwen2.5-VL's special block requirements
            # The post-projection LayerNorm will provide stability
            vision_features = self.vision_model(pixel_values, grid_thw=image_grid_thw)

        # Keep in bf16 for memory efficiency
        vision_features = vision_features.to(dtype=torch.bfloat16)

        # Clip vision features to prevent extreme values
        vision_features = torch.clamp(vision_features, min=-10.0, max=10.0)

        # Check if vision model output is NaN
        if torch.isnan(vision_features).any():
            logger.warning(f"Vision model output contains NaN!")
            logger.warning(f"  Shape: {vision_features.shape}")
            # Don't replace with zeros - let's see what happens
            # vision_features = torch.zeros_like(vision_features)

        # Check for NaN and clip if needed
        if torch.isnan(vision_features).any() or torch.isinf(vision_features).any():
            logger.warning(f"Vision features contain NaN/Inf!")
            logger.warning(f"  Max before fix: {vision_features.abs().max().item() if not torch.isnan(vision_features).all() else 'all nan'}")
            vision_features = torch.nan_to_num(vision_features, nan=0.0, posinf=1.0, neginf=-1.0)
            # Also clip to reasonable range
            vision_features = torch.clamp(vision_features, min=-10.0, max=10.0)
            logger.warning(f"  Max after fix: {vision_features.abs().max().item()}")

        # vision_features shape: [total_patches, qwen3_hidden_dim] after our replaced merger

        # Cache for inference (quick and dirty)
        if not self.training:
            self._vision_cache = vision_features
            logger.debug("Cached vision embeddings for reuse")

        return vision_features

    def forward(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor] = None,
                image_grid_thw: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[tuple] = None,
                use_cache: Optional[bool] = None,
                **kwargs):

        # === Decoding Phase (KV Cache Hit) ===
        # If we have a cache, we bypass vision processing and embedding merging.
        if past_key_values is not None:
            # We are decoding. input_ids should typically be just the last token [B, 1].
            # Use the language model directly.
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.language_model(
                    input_ids=input_ids,  # Only the last token
                    attention_mask=attention_mask,
                    labels=labels,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_hidden_states=False,
                    return_dict=True,
                )
            return outputs

        # === Prefill Phase (or Training) ===
        # Use get_input_embeddings to ensure we get the right embedding layer
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)


        # Debug gradient status - commented out now that it's working
        # if not hasattr(self, '_forward_debug'):
        #     self._forward_debug = True
        #     print(f"\nDEBUG Forward Pass:")
        #     print(f"  text_embeddings requires_grad: {text_embeddings.requires_grad}")
        #     print(f"  text_embeddings shape: {text_embeddings.shape}")
        #     print(f"  vision_start_embedding requires_grad: {self.vision_start_embedding.requires_grad}")
        #     print(f"  vision_end_embedding requires_grad: {self.vision_end_embedding.requires_grad}")
        #     print(f"  input_ids shape: {input_ids.shape}")
        #     print(f"  Has pixel_values: {pixel_values is not None}")
        #     if pixel_values is not None:
        #         print(f"  pixel_values shape: {pixel_values.shape}")
        #         print(f"  image_grid_thw shape: {image_grid_thw.shape}")

        if pixel_values is None:
            # Text-only path: use standard language model forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.language_model(
                    inputs_embeds=text_embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=use_cache if use_cache is not None else False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            
            # Log text-only loss for debugging - commented out for cleaner output
            # if labels is not None and outputs.loss is not None:
            #     valid_tokens = (labels != -100).sum().item()
            #     logger.debug(f"[TEXT-ONLY LOSS] HF loss={outputs.loss.item():.4f}, valid_tokens={valid_tokens}")
            
            return outputs

        # Avoid stale cache during evaluation (eval typically passes labels)
        if (not self.training) and (labels is not None):
            self.clear_vision_cache()

        # Get vision embeddings
        vision_embeddings = self.get_vision_embeddings(pixel_values, image_grid_thw)

        # Apply post-projection LayerNorm for stability
        vision_embeddings = self.post_proj_ln(vision_embeddings)

        # Split vision embeddings by image based on grid_thw
        split_sizes = (image_grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()

        # Debug: Check for size mismatch
        total_expected = sum(split_sizes)
        actual_size = vision_embeddings.shape[0]
        if total_expected != actual_size:
            logger.warning(f"Vision embedding size mismatch: expected {total_expected}, got {actual_size}")
            logger.warning(f"  image_grid_thw: {image_grid_thw.tolist()}")
            logger.warning(f"  split_sizes: {split_sizes}")
            logger.warning(f"  spatial_merge_size: {self.spatial_merge_size}")
            # Adjust split sizes to match actual embeddings
            if len(split_sizes) == 1:
                # Single image - just use all embeddings
                split_sizes = [actual_size]
            else:
                # Multiple images - scale proportionally
                scale = actual_size / total_expected
                split_sizes = [int(s * scale) for s in split_sizes]
                # Adjust last to match exactly
                split_sizes[-1] = actual_size - sum(split_sizes[:-1])
            logger.warning(f"  Adjusted split_sizes: {split_sizes}")

        vision_embeds_list = torch.split(vision_embeddings, split_sizes)

        new_input_embeds, new_labels, new_attention_mask = [], [], []
        placeholder_mask = (input_ids == self.config.vision_placeholder_id)

        # More debug - commented out now that it's working
        # if hasattr(self, '_forward_debug'):
        #     print(f"\nDEBUG Placeholder Processing:")
        #     print(f"  Vision placeholder id: {self.config.vision_placeholder_id}")
        #     print(f"  Placeholder mask shape: {placeholder_mask.shape}")
        #     print(f"  Placeholder mask sum: {placeholder_mask.sum().item()}")
        #     print(f"  Batch size: {input_ids.shape[0]}")
        #     # Check each sample
        #     for i in range(min(2, input_ids.shape[0])):
        #         sample_placeholders = placeholder_mask[i].sum().item()
        #         print(f"  Sample {i}: {sample_placeholders} placeholders")
        #         if sample_placeholders > 0:
        #             # Show where placeholders are
        #             indices = torch.where(placeholder_mask[i])[0]
        #             print(f"    Placeholder indices: {indices.tolist()[:5]}...")  # First 5

        for i in range(input_ids.shape[0]):
            placeholder_indices = torch.where(placeholder_mask[i])[0]
            if not len(placeholder_indices):
                new_input_embeds.append(text_embeddings[i])
                if labels is not None: new_labels.append(labels[i])
                new_attention_mask.append(attention_mask[i])
                continue

            # For simplicity, assume one image per sample
            marker_pos = placeholder_indices[0]
            pre_marker_embeds = text_embeddings[i, :marker_pos]
            post_marker_embeds = text_embeddings[i, marker_pos + 1:]

            # Get the vision embeddings for this sample
            vis = vision_embeds_list[i]  # [T, D] projected+LN'd

            if self.disable_pooling:
                # Pass all vision tokens directly without any pooling/sampling
                vision_embeds = vis
            else:
                # ===== Token curriculum with adjustable cap =====
                # Build: [global, base_k pooled tokens, optional extra tokens]
                global_tok = self._global_token(vis)  # [1, D]
                base_k = min(self.base_k, self.max_vision_tokens - 1)  # Leave room for global token
                base_pooled = self._resample_tokens(vis, k=base_k)  # [base_k, D]
                
                # Only use extra tokens if curriculum allows and we have budget
                use_extras = self.training and self.allow_extra_tokens
                if use_extras:
                    extra_budget = self.max_vision_tokens - (1 + base_pooled.shape[0])
                    if extra_budget > 0:
                        # Sample extra tokens evenly from the vision features
                        idx = torch.linspace(0, vis.shape[0] - 1, steps=extra_budget, device=vis.device).long()
                        extra_tokens = vis.index_select(0, idx)
                        if self.use_soft_gate:
                            extra_tokens = torch.tanh(self.extra_gate) * extra_tokens
                        vision_embeds = torch.cat([global_tok, base_pooled, extra_tokens], dim=0)
                    else:
                        vision_embeds = torch.cat([global_tok, base_pooled], dim=0)
                else:
                    # No extras - just global + base pooled
                    vision_embeds = torch.cat([global_tok, base_pooled], dim=0)
                # =============================
            
            # Log vision token usage (first few steps only for debugging)
            if not hasattr(self, '_vision_token_log_count'):
                self._vision_token_log_count = 0
            if self._vision_token_log_count < 5:
                if self.disable_pooling:
                    logger.info(f"[seq] vision_tokens={vision_embeds.shape[0]} (all tokens, no pooling)")
                else:
                    logger.info(f"[seq] vision_tokens={vision_embeds.shape[0]} max_cap={self.max_vision_tokens} base_k={self.base_k} extras={'on' if self.allow_extra_tokens else 'off'}")
                self._vision_token_log_count += 1

            # Debug vision embedding - commented out now that it's working
            # if i < 2 and hasattr(self, '_forward_debug'):
            #     print(f"\n  Sample {i} vision insertion:")
            #     print(f"    Vision embeds shape: {vision_embeds.shape}")
            #     print(f"    Vision embeds requires_grad: {vision_embeds.requires_grad}")
            #     print(f"    Marker position: {marker_pos}")
            #     print(f"    Pre-marker length: {pre_marker_embeds.shape[0]}")
            #     print(f"    Post-marker length: {post_marker_embeds.shape[0]}")

            # Ensure vision embeddings are on the same device and dtype
            vision_start = self.vision_start_embedding.unsqueeze(0).to(pre_marker_embeds.device, pre_marker_embeds.dtype)
            vision_end = self.vision_end_embedding.unsqueeze(0).to(pre_marker_embeds.device, pre_marker_embeds.dtype)
            vision_embeds = vision_embeds.to(pre_marker_embeds.device, pre_marker_embeds.dtype)

            # Check vision embedding scale before using
            if hasattr(self, '_debug_loss_counter') and self._debug_loss_counter < 5:
                v_max = vision_embeds.abs().max().item()
                v_mean = vision_embeds.abs().mean().item()
                t_max = pre_marker_embeds.abs().max().item() if pre_marker_embeds.shape[0] > 0 else 0
                t_mean = pre_marker_embeds.abs().mean().item() if pre_marker_embeds.shape[0] > 0 else 0
                logger.debug(f"    Vision embeds: max={v_max:.4f}, mean={v_mean:.4f}")
                logger.debug(f"    Text embeds: max={t_max:.4f}, mean={t_mean:.4f}")

            # No manual scaling - the trained adapter handles this

            combined_embeds = torch.cat([
                pre_marker_embeds,
                vision_start,
                vision_embeds,
                vision_end,
                post_marker_embeds,
            ], dim=0)


            new_input_embeds.append(combined_embeds)

            if labels is not None:
                pre_marker_labels = labels[i, :marker_pos]
                post_marker_labels = labels[i, marker_pos + 1:]
                vision_part_len = 1 + vision_embeds.shape[0] + 1
                vision_labels = torch.full((vision_part_len,), -100, device=labels.device, dtype=torch.long)
                combined_labels = torch.cat([pre_marker_labels, vision_labels, post_marker_labels], dim=0)
                new_labels.append(combined_labels)

            pre_marker_mask = attention_mask[i, :marker_pos]
            post_marker_mask = attention_mask[i, marker_pos + 1:]
            vision_mask = torch.ones(1 + vision_embeds.shape[0] + 1, device=attention_mask.device, dtype=torch.long)
            combined_mask = torch.cat([pre_marker_mask, vision_mask, post_marker_mask], dim=0)
            new_attention_mask.append(combined_mask)

        padded_embeds = torch.nn.utils.rnn.pad_sequence(new_input_embeds, batch_first=True, padding_value=0.0)
        padded_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
        padded_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=-100) if labels is not None else None

        # Final debug check - commented out now that it's working
        # if hasattr(self, '_forward_debug'):
        #     print(f"\nDEBUG Final tensors:")
        #     print(f"  Padded embeds shape: {padded_embeds.shape}")
        #     print(f"  Padded embeds requires_grad: {padded_embeds.requires_grad}")
        #     print(f"  Number of samples with vision: {len(new_input_embeds)}")
        #     # Check if any embedding requires grad
        #     any_grad = any(e.requires_grad for e in new_input_embeds if isinstance(e, torch.Tensor))
        #     print(f"  Any embedding requires grad: {any_grad}")
        #     self._forward_debug = False  # Only print once


        # Debug loss calculation
        if hasattr(self, '_debug_loss_counter'):
            self._debug_loss_counter += 1
        else:
            self._debug_loss_counter = 0

        if self._debug_loss_counter < 5 and padded_labels is not None:
            # Check label distribution
            non_masked = (padded_labels != -100).sum().item()
            total = padded_labels.numel()
            logger.debug(f"Loss calculation (sample {self._debug_loss_counter}):")
            logger.debug(f"  Labels shape: {padded_labels.shape}")
            logger.debug(f"  Non-masked tokens: {non_masked}/{total} ({non_masked/total*100:.1f}%)")
            logger.debug(f"  Padded embeds requires_grad: {padded_embeds.requires_grad}")

            # Check for NaN/Inf in embeddings
            has_nan = torch.isnan(padded_embeds).any().item()
            has_inf = torch.isinf(padded_embeds).any().item()
            embed_max = padded_embeds.abs().max().item()
            embed_mean = padded_embeds.abs().mean().item()
            logger.debug(f"  Embeddings: has_nan={has_nan}, has_inf={has_inf}, max={embed_max:.4f}, mean={embed_mean:.4f}")

        # Log sequence stats for debugging - commented out for cleaner output
        B, T, D = padded_embeds.shape
        # logger.debug(f"[seq stats] B={B}, T={T}, D={D} (vision<=128, text<=768 expected)")
        
        # Use the standard language model forward pass with Flash Attention
        # This will compute loss internally using the standard HF implementation
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.language_model(
                inputs_embeds=padded_embeds,
                attention_mask=padded_mask,
                labels=padded_labels,
                use_cache=use_cache if use_cache is not None else False,
                output_hidden_states=False,
                return_dict=True,
            )
        
        # Log the loss value for debugging - commented out for cleaner output
        # if padded_labels is not None and outputs.loss is not None:
        #     valid_tokens = (padded_labels != -100).sum().item()
        #     logger.debug(f"[LOSS] HF loss={outputs.loss.item():.4f}, valid_tokens={valid_tokens}")
        
        return outputs

    def save_pretrained(self, save_directory: str, **kwargs):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save the configuration (saves as config.json)
        self.config.save_pretrained(save_directory)
        
        # 2. Save trainable state dict (vision_adapter.pt)
        trainable_state_dict = {
            'vision_merger_state': self.vision_model.merger.state_dict(),
            'vision_start_embedding': self.vision_start_embedding,
            'vision_end_embedding': self.vision_end_embedding,
            'post_proj_ln_state': self.post_proj_ln.state_dict(),
            'extra_gate': self.extra_gate,
        }
        torch.save(trainable_state_dict, path / "vision_adapter.pt")

        # 3. Save the tokenizer
        if hasattr(self, 'tokenizer') and self.tokenizer:
            self.tokenizer.save_pretrained(save_directory)
            
        logger.info(f"Configuration and adapter weights saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)

        # 1. Load the configuration
        # Handle legacy checkpoints (adapter_config.json) vs new checkpoints (config.json)
        config_file = "config.json"
        if not (path / "config.json").exists() and (path / "adapter_config.json").exists():
            logger.warning("Loading legacy checkpoint (adapter_config.json). Please resave checkpoint to update to config.json.")
            # Load legacy config and convert to new format
            with open(path / "adapter_config.json", "r") as f:
                legacy_config = json.load(f)
            config = QwenQwenHybridConfig(
                vision_model_name=legacy_config.get('vision_model_name', "Qwen/Qwen2.5-VL-7B-Instruct"),
                language_model_name=legacy_config.get('language_model_name', "Qwen/Qwen3-4B"),
                vision_placeholder_id=legacy_config.get('vision_placeholder_id'),
                vision_start_token_id=legacy_config.get('vision_start_token_id'),
                vision_end_token_id=legacy_config.get('vision_end_token_id'),
            )
        else:
            # Load the config using the standard HF mechanism
            config = QwenQwenHybridConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 2. Initialize the model structure (loads base models)
        model = cls(config)

        # 3. Load the custom adapter weights
        adapter_weights_path = path / "vision_adapter.pt"
        if not adapter_weights_path.exists():
            raise FileNotFoundError(f"Adapter weights (vision_adapter.pt) not found in {pretrained_model_name_or_path}")

        adapter_state = torch.load(adapter_weights_path, map_location="cpu")
        model.vision_model.merger.load_state_dict(adapter_state['vision_merger_state'])
        model.vision_start_embedding.data.copy_(adapter_state['vision_start_embedding'].data)
        model.vision_end_embedding.data.copy_(adapter_state['vision_end_embedding'].data)
        
        # Load post_proj_ln if present
        if 'post_proj_ln_state' in adapter_state:
            model.post_proj_ln.load_state_dict(adapter_state['post_proj_ln_state'])
            logger.info(f"Loaded post_proj_ln from checkpoint")
        else:
            logger.warning("post_proj_ln_state missing in checkpoint - using fresh initialization (expect loss spike)")
        
        # Load extra_gate if present
        if 'extra_gate' in adapter_state:
            model.extra_gate.data.copy_(adapter_state['extra_gate'].data)
            logger.info(f"Loaded extra_gate={adapter_state['extra_gate'].item():.3f} from checkpoint")
        else:
            logger.warning("extra_gate missing in checkpoint - using fresh initialization")
        
        # Log sanity check values
        logger.info(f"post_proj_ln gamma mean={model.post_proj_ln.weight.mean().item():.4f}")
        logger.info(f"post_proj_ln beta mean={model.post_proj_ln.bias.mean().item():.4f}")
        logger.info(f"extra_gate={model.extra_gate.item():.3f}")

        # 4. Load LoRA adapter if it exists
        lora_path = path / "lora_adapter"
        if lora_path.exists():
            from peft import PeftModel
            model.language_model = PeftModel.from_pretrained(model.language_model, str(lora_path))
            logger.info(f"Loaded LoRA adapter from {lora_path}")

        logger.info(f"Loaded hybrid model from {pretrained_model_name_or_path}")
        return model

# Register with AutoModel
AutoConfig.register("qwen_qwen_hybrid", QwenQwenHybridConfig)
AutoModelForCausalLM.register(QwenQwenHybridConfig, QwenQwenHybrid)

# --- Test ---
if __name__ == "__main__":
    # Update the test block to use the new helper method
    model = QwenQwenHybrid.from_base_models()
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    logger.info(f"Total trainable parameters: {trainable_params:,}")
    model.save_pretrained("./qwen_r1_finished_hybrid")
    loaded_model = QwenQwenHybrid.from_pretrained("./qwen_r1_finished_hybrid")
    logger.info("Model saved and loaded successfully!")
