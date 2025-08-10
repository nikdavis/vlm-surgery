#!/usr/bin/env python3
"""
Inference for Qwen2.5-VL vision + Qwen3 LLM hybrid without HF.generate().

Example:
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run python -m src.surgery.inference \
  --checkpoint outputs_qwen_hybrid/checkpoint-1500 \
  --image cot/example_0189/problem_image.jpg \
  --prompt "In this image, I can see" \
  --max-tokens 50 \
  --temperature 0.2
"""
from pathlib import Path
import click
import torch
from PIL import Image
from loguru import logger
from transformers import AutoProcessor
from src.surgery.qwen_qwen_vision import QwenQwenHybrid

system_prompt = """
You are a guessing engine.
You will guess exactly one object that appears in the provided image context tokens.
Do not explain. Do not mention uncertainty.
Output only the object's name as a single word.
"""


def _find_user_insert_pos(tokenizer, ids):
    """Insert after <｜User｜> if present; fallback to legacy prefix."""
    user_tok_id = tokenizer.convert_tokens_to_ids("<｜User｜>")
    if user_tok_id is not None and user_tok_id != tokenizer.unk_token_id:
        for j in range(len(ids) - 1):
            if ids[j] == user_tok_id:
                return j + 1
    legacy = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    if legacy:
        return len(legacy)
    return min(1, len(ids))


def _extract_assistant_span(full_text: str) -> str:
    if "<|im_start|>assistant" in full_text:
        part = full_text.split("<|im_start|>assistant", 1)[-1]
        if "<|im_end|>" in part:
            part = part.split("<|im_end|>", 1)[0]
        return part.strip()
    return full_text.strip()


@click.command()
@click.option("--checkpoint", required=True, help="Adapter dir containing adapter_config.json & vision_adapter.pt")
@click.option("--image", required=True, type=str, help="Image path")
@click.option("--prompt", required=True, type=str, help="User prompt")
@click.option("--system", default=system_prompt)
@click.option("--max-tokens", default=128, type=int)
@click.option("--num-shots", default=10, type=int, help="Number of independent generations to produce.")
@click.option("--temperature", default=0.4, type=float)
@click.option("--top-p", default=0.9, type=float)
@click.option("--do-sample/--no-sample", default=True)
def main(checkpoint, image, prompt, system, max_tokens, num_shots, temperature, top_p, do_sample):
    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

    ckpt = Path(checkpoint)
    imgp = Path(image)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if not imgp.exists():
        raise FileNotFoundError(imgp)

    logger.info(f"Loading hybrid model from adapters: {ckpt}")
    model = QwenQwenHybrid.from_pretrained(str(ckpt)).cuda().eval()
    if hasattr(model, "clear_vision_cache"):
        model.clear_vision_cache()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    pil = Image.open(imgp).convert("RGB")
    img_inputs = processor.image_processor([pil], return_tensors="pt")
    pixel_values = img_inputs.pixel_values.cuda()
    image_grid_thw = img_inputs.image_grid_thw.cuda()

    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Chat prefix using template; add_generation_prompt=True adds assistant header
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text_prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # Insert exactly one <|image_pad|> after the user tag
    ids = tok.encode(text_prefix, add_special_tokens=False)
    insert_pos = _find_user_insert_pos(tok, ids)
    image_pad_id = model.config.vision_placeholder_id
    ids = ids[:insert_pos] + [image_pad_id] + ids[insert_pos:]

    initial_input_ids = torch.tensor([ids], device=pixel_values.device)
    initial_attn = torch.ones_like(initial_input_ids, device=initial_input_ids.device)

    # Stop on EOS or <|im_end|>
    stop_ids = []
    if tok.eos_token_id is not None:
        stop_ids.append(tok.eos_token_id)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None:
        stop_ids.append(im_end_id)
    stop_ids = set(stop_ids)

    logger.info("Generating...")
    for i in range(num_shots):
        if num_shots > 1:
            logger.info(f"\n--- Shot {i+1}/{num_shots} ---\n")

        input_ids = initial_input_ids.clone()
        attn = initial_attn.clone()

        with torch.inference_mode():
            for step in range(max_tokens):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=None,
                )
                logits = out.logits if hasattr(out, "logits") else out
                next_logits = logits[:, -1, :]  # [B,V]

                if do_sample:
                    # temperature + top-p
                    temp = temperature
                    if temp is None or temp <= 0:
                        temp = 1.0
                    next_logits = next_logits / temp
                    probs = torch.softmax(next_logits, dim=-1)

                    if top_p is not None and 0 < top_p < 1.0:
                        # nucleus sampling
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cdf = torch.cumsum(sorted_probs, dim=-1)
                        mask = cdf > top_p
                        # keep at least 1 token
                        mask[..., 0] = False
                        probs = probs.scatter(1, sorted_idx, sorted_probs.masked_fill(mask, 0.0))
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                    next_token = torch.multinomial(probs, 1)  # [B,1]
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B,1]

                input_ids = torch.cat([input_ids, next_token], dim=1)
                attn = torch.ones_like(input_ids, device=input_ids.device)

                tid = int(next_token.item())
                if tid in stop_ids:
                    break

        decoded = tok.decode(input_ids[0], skip_special_tokens=False)
        assistant = _extract_assistant_span(decoded)
        print(assistant.strip())


if __name__ == "__main__":
    main()
