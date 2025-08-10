"""Tests for src.surgery.train_qwen_qwen module."""

import pytest
import torch
from transformers import AutoTokenizer, AutoProcessor
import json
import numpy as np
from pathlib import Path


@pytest.fixture
def tokenizer():
    """Fixture for Qwen tokenizer."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


@pytest.fixture
def processor():
    """Fixture for Qwen processor."""
    return AutoProcessor.from_pretrained("Qwen2/Qwen2.5-VL-1.5B")


@pytest.fixture
def collator(tokenizer, processor):
    """Fixture for CoTDataCollator."""
    from src.surgery.train_qwen_qwen import CoTDataCollator
    
    return CoTDataCollator(
        tokenizer=tokenizer,
        image_processor=processor.image_processor
    )


class TestTokenizer:
    """Test tokenizer functionality."""
    
    def test_special_tokens(self, tokenizer):
        """Test that special tokens are properly recognized."""
        special_tokens = {
            "<|im_start|>": tokenizer.convert_tokens_to_ids("<|im_start|>"),
            "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "<|vision_start|>": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "<|vision_end|>": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            "<|image_pad|>": tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        }
        
        # All special tokens should have valid IDs
        for name, token_id in special_tokens.items():
            assert token_id is not None, f"Special token {name} not found"
            assert token_id >= 0, f"Special token {name} has invalid ID"
    
    def test_chat_template(self, tokenizer):
        """Test chat template formatting."""
        messages = [
            {"role": "user", "content": "What is in this image?"},
            {"role": "assistant", "content": "This is a test response."}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Should contain both user and assistant markers
        assert "<|im_start|>user" in text
        assert "<|im_start|>assistant" in text
        assert "<|im_end|>" in text
    
    def test_assistant_response_extraction(self, tokenizer):
        """Test that we can find and extract assistant response."""
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "This is the answer."}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Find assistant marker
        assistant_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        assistant_pos = None
        
        for i in range(len(tokens) - len(assistant_marker) + 1):
            if tokens[i:i+len(assistant_marker)] == assistant_marker:
                assistant_pos = i + len(assistant_marker)
                break
        
        assert assistant_pos is not None, "Could not find assistant marker"
        
        # Find end token
        end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
        end_pos = None
        for i in range(assistant_pos, len(tokens)):
            if tokens[i] == end_token:
                end_pos = i
                break
        
        assert end_pos is not None, "Could not find end token"
        
        # Extract and verify assistant response
        assistant_tokens = tokens[assistant_pos:end_pos]
        assistant_text = tokenizer.decode(assistant_tokens)
        
        assert "This is the answer" in assistant_text
        assert len(assistant_tokens) > 0


class TestCoTDataCollator:
    """Test CoTDataCollator functionality."""
    
    def test_basic_collation(self, collator):
        """Test basic collation without images."""
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What do you see?"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I see a beautiful landscape with mountains."}]
                }
            ],
            "images": []
        }
        
        batch = collator([sample])
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        
        # Check shapes match
        assert batch["input_ids"].shape == batch["labels"].shape
        assert batch["input_ids"].shape == batch["attention_mask"].shape
    
    def test_label_masking(self, collator, tokenizer):
        """Test that labels are properly masked."""
        assistant_text = "This is the assistant response that should be trained on."
        
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question?"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}]
                }
            ],
            "images": []
        }
        
        batch = collator([sample])
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]
        
        # Should have some unmasked tokens
        unmasked_count = (labels != -100).sum().item()
        assert unmasked_count > 0, "No unmasked tokens - nothing to train on!"
        
        # Unmasked tokens should correspond to assistant response
        unmasked_indices = (labels != -100).nonzero(as_tuple=True)[0]
        unmasked_text = tokenizer.decode(input_ids[unmasked_indices].tolist())
        
        # The unmasked text should contain most of the assistant response
        # (might not be exact due to tokenization)
        assert len(unmasked_text) > 0
        
        # Should NOT train on control tokens
        control_tokens = ["<|im_start|>", "<|im_end|>", "<|im_start|>assistant"]
        for control in control_tokens:
            control_id = tokenizer.convert_tokens_to_ids(control)
            if control_id:
                assert (labels == control_id).sum().item() == 0, f"Training on control token: {control}"
    
    def test_causal_shift_compatibility(self, collator):
        """Test that labels work with causal shift."""
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Q?"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "A long answer with many tokens to ensure we have enough for training."}]
                }
            ],
            "images": []
        }
        
        batch = collator([sample])
        labels = batch["labels"][0]
        
        # Check tokens available after shift
        valid_before = (labels != -100).sum().item()
        valid_after = (labels[1:] != -100).sum().item() if len(labels) > 1 else 0
        
        assert valid_after > 5, f"Only {valid_after} tokens after shift - too few for reliable training"
        
        # Most tokens should survive the shift
        if valid_before > 0:
            retention_rate = valid_after / valid_before
            assert retention_rate > 0.8, f"Lost too many tokens in shift: {retention_rate:.1%}"
    
    def test_with_images(self, collator):
        """Test collation with images."""
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What's in the image?"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I see an image."}]
                }
            ],
            "images": [dummy_image]
        }
        
        batch = collator([sample])
        
        # Should have image data
        assert "pixel_values" in batch
        assert batch["pixel_values"] is not None
        
        # Should have image placeholders in input_ids
        input_ids = batch["input_ids"][0]
        placeholder_id = collator.placeholder_id
        placeholder_count = (input_ids == placeholder_id).sum().item()
        
        assert placeholder_count > 0, "No image placeholders found in input"
    
    def test_batch_collation(self, collator):
        """Test collating multiple samples."""
        samples = [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": f"Question {i}?"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": f"Answer {i}."}]}
                ],
                "images": []
            }
            for i in range(3)
        ]
        
        batch = collator(samples)
        
        # Should have batch dimension
        assert batch["input_ids"].shape[0] == 3
        assert batch["labels"].shape[0] == 3
        assert batch["attention_mask"].shape[0] == 3
        
        # Each sample should have some unmasked tokens
        for i in range(3):
            unmasked = (batch["labels"][i] != -100).sum().item()
            assert unmasked > 0, f"Sample {i} has no unmasked tokens"


class TestRealDataIntegration:
    """Test with real dataset if available."""
    
    @pytest.mark.skipif(
        not any(Path(p).exists() for p in ["data-captions/dataset_*/*.parquet", "data-captions/*.parquet"]),
        reason="No parquet files found"
    )
    def test_real_sample_tokenization(self, tokenizer):
        """Test tokenization with real dataset sample."""
        import glob
        import pyarrow.parquet as pq
        
        # Find parquet files
        parquet_files = glob.glob("data-captions/dataset_*/*.parquet") or glob.glob("data-captions/*.parquet")
        
        if parquet_files:
            # Read first sample
            table = pq.read_table(parquet_files[0])
            df = table.to_pandas()
            
            if len(df) > 0:
                sample = df.iloc[0]
                
                if 'prompts' in sample:
                    prompts = json.loads(sample['prompts']) if isinstance(sample['prompts'], str) else sample['prompts']
                    if prompts and len(prompts) > 0:
                        prompt_text = prompts[0]
                        
                        # Tokenize the prompt
                        tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
                        
                        # Should have reasonable length
                        assert len(tokens) > 10, f"Prompt too short: only {len(tokens)} tokens"
                        assert len(tokens) < 2000, f"Prompt too long: {len(tokens)} tokens"