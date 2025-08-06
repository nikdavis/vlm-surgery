# Corrected version for: /home/nik/code/langgraph-work-dir/fine-pixtral/src/surgery/hybrid_model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from typing import Dict, Tuple, Optional

class InternVL3VisionAdapter(nn.Module):
    """
    Adapter for InternVL3 vision encoder -> DeepSeek R1.
    Handles: 3200 dims -> 4096 dims.
    This version does NOT compress the sequence of vision tokens.
    """
    def __init__(self, vision_hidden=3200, language_hidden=4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(vision_hidden),
            nn.Linear(vision_hidden, language_hidden * 2),
            nn.GELU(),
            nn.Linear(language_hidden * 2, language_hidden),
            nn.LayerNorm(language_hidden)
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # Project every vision token to the language dimension without compression.
        return self.projection(vision_features)

class EnhancedVisionMerger:
    # ... (This class is correct and does not need changes) ...
    """Handles the merging of vision embeddings into the text sequence."""
    def __init__(self, start_id=20006):
        self.start_id = start_id

    def merge_vision_with_text(self, input_ids: torch.Tensor, text_embeddings: torch.Tensor,
                              vision_embeddings: torch.Tensor, end_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, device = text_embeddings.shape[0], text_embeddings.device
        start_masks = (input_ids == self.start_id)
        merged_sequences, attention_masks = [], []

        for b in range(batch_size):
            start_positions = start_masks[b].nonzero(as_tuple=True)[0]
            if len(start_positions) == 0:
                merged_sequences.append(text_embeddings[b])
                attention_masks.append(torch.ones(text_embeddings.shape[1], device=device))
                continue

            parts, attn_parts, last_pos = [], [], 0
            vision_per_image = vision_embeddings.shape[1] // len(start_positions)

            for i, start_pos in enumerate(start_positions):
                start_pos = start_pos.item()
                parts.append(text_embeddings[b, last_pos:start_pos + 1])
                attn_parts.append(torch.ones(start_pos + 1 - last_pos, device=device))

                vision_start, vision_end = i * vision_per_image, (i + 1) * vision_per_image
                parts.append(vision_embeddings[b, vision_start:vision_end])
                attn_parts.append(torch.ones(vision_per_image, device=device))

                parts.append(end_embed.unsqueeze(0))
                attn_parts.append(torch.ones(1, device=device))
                last_pos = start_pos + 1

            if last_pos < text_embeddings.shape[1]:
                parts.append(text_embeddings[b, last_pos:])
                attn_parts.append(torch.ones(text_embeddings.shape[1] - last_pos, device=device))

            merged_sequences.append(torch.cat(parts, dim=0))
            attention_masks.append(torch.cat(attn_parts, dim=0))

        max_len = max(seq.shape[0] for seq in merged_sequences)
        padded_sequences, padded_masks = [], []

        for seq, mask in zip(merged_sequences, attention_masks):
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                seq = torch.cat([seq, torch.zeros(pad_len, seq.shape[1], device=device, dtype=seq.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, device=device, dtype=mask.dtype)])
            padded_sequences.append(seq)
            padded_masks.append(mask)

        return torch.stack(padded_sequences), torch.stack(padded_masks)


# In: /home/nik/code/langgraph-work-dir/fine-pixtral/src/surgery/hybrid_model.py

# ... (InternVL3VisionAdapter class remains the same) ...

class InternVL3DeepSeekR1Hybrid(nn.Module):
    """
    Complete implementation: InternVL3 Vision + DeepSeek R1 Language.
    This version uses a simplified forward pass for better Hugging Face Trainer compatibility.
    """
    def __init__(self, vision_model_name="OpenGVLab/InternVL3-78B", language_model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        super().__init__()
        print(f"Loading LLM: {language_model_name}")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.language_model.config.use_cache = False # Required for gradient checkpointing
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name, trust_remote_code=True)

        print(f"Loading Vision Encoder from: {vision_model_name}")
        self.vision_encoder = self._load_vision_encoder(vision_model_name)

        print("Creating Vision-Language Adapter...")
        self.vision_adapter = InternVL3VisionAdapter(vision_hidden=3200, language_hidden=4096)
        self.vision_adapter.to(self.language_model.dtype)

        self.image_marker_id = 20006 # ' unused'

        self._freeze_models()
        print("Hybrid model ready.")

    def _load_vision_encoder(self, model_name: str):
        intern_vl_model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True
        )
        vision_encoder = intern_vl_model.vision_model
        del intern_vl_model
        return vision_encoder

    def _freeze_models(self):
        for param in self.language_model.parameters(): param.requires_grad = False
        for param in self.vision_encoder.parameters(): param.requires_grad = False
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        # If there are no images, just pass through to the language model
        if pixel_values is None:
            return self.language_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # 1. Get text and vision embeddings
        text_embeddings = self.language_model.model.embed_tokens(input_ids)
        vision_features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        vision_embeddings = self.vision_adapter(vision_features)

        # 2. Build the new sequence by replacing the image marker
        batch_size = input_ids.shape[0]
        new_input_embeds = []
        new_labels = []
        new_attention_mask = []

        for i in range(batch_size):
            # Find the position of the image marker token
            marker_indices = (input_ids[i] == self.image_marker_id).nonzero(as_tuple=True)[0]
            if len(marker_indices) == 0: continue # Should not happen if data collator is correct
            marker_pos = marker_indices[0]

            # Split embeddings and labels around the marker
            pre_marker_embeds = text_embeddings[i, :marker_pos]
            post_marker_embeds = text_embeddings[i, marker_pos + 1:]

            # Combine to form the full embedding sequence
            combined_embeds = torch.cat([pre_marker_embeds, vision_embeddings[i], post_marker_embeds], dim=0)
            new_input_embeds.append(combined_embeds)

            # Do the same for labels
            if labels is not None:
                pre_marker_labels = labels[i, :marker_pos]
                # Create vision labels filled with -100
                vision_labels = torch.full((vision_embeddings.shape[1],), -100, device=labels.device)
                post_marker_labels = labels[i, marker_pos + 1:]

                combined_labels = torch.cat([pre_marker_labels, vision_labels, post_marker_labels], dim=0)
                new_labels.append(combined_labels)

            # Do the same for the attention mask
            pre_marker_mask = attention_mask[i, :marker_pos]
            vision_mask = torch.ones(vision_embeddings.shape[1], device=attention_mask.device)
            post_marker_mask = attention_mask[i, marker_pos + 1:]

            combined_mask = torch.cat([pre_marker_mask, vision_mask, post_marker_mask], dim=0)
            new_attention_mask.append(combined_mask)

        # 3. Pad the batch to the longest sequence
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_input_embeds, batch_first=True, padding_value=0.0)
        final_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=-100) if labels is not None else None
        final_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

        # 4. Pass everything to the language model
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            **kwargs
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        print("Enabling gradient checkpointing on the language model...")
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def save_pretrained(self, save_directory: str):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "vision_model_name": "OpenGVLab/InternVL3-78B",
            "language_model_name": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        }
        with open(path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        torch.save({
            'vision_adapter_state_dict': self.vision_adapter.state_dict(),
            'start_embed': self.start_embed,
            'end_embed': self.end_embed,
        }, path / "adapter_model.pt")
        self.tokenizer.save_pretrained(save_directory)
        print(f"Trainable components saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str):
        # ...
        path = Path(load_directory)
        with open(path / "adapter_config.json", "r") as f:
            config = json.load(f)
        model = cls(
            vision_model_name=config["vision_model_name"],
            language_model_name=config["language_model_name"]
        )
        saved_state = torch.load(path / "adapter_model.pt", map_location="cpu")
        model.vision_adapter.load_state_dict(saved_state['vision_adapter_state_dict'])
        model.start_embed.data = saved_state['start_embed'].data
        model.end_embed.data = saved_state['end_embed'].data
        print(f"Trainable components loaded from {load_directory}")
        return model
