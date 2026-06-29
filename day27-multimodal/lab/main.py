"""
Day 27: Multimodal Inference — Vision-Language Models

Process both images and text through a vision-language model to see how
multimodal inputs are tokenized, how image resolution affects token count,
and the speed overhead of visual processing vs text-only inference.

Model: Qwen/Qwen2.5-VL-7B-Instruct (Apache 2.0, ungated)
"""

import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --- Load the vision-language model ---
# Qwen2.5-VL-7B-Instruct handles both text and images. The processor
# converts images into visual tokens that the model can attend to alongside
# text tokens. device_map="auto" places layers on GPU if available.
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

print("=" * 70)
print("MULTIMODAL INFERENCE: Vision-Language Model")
print("=" * 70)
print(f"\nLoading model: {MODEL_NAME}")
print("(First run downloads ~15 GB — subsequent runs use cache)\n")

try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map="auto"
    )
except OSError as e:
    print(f"Failed to load model: {e}")
    print("Check your internet connection and disk space, then try again.")
    raise SystemExit(1)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {param_count / 1e9:.1f}B parameters")
print(f"Device: {model.device}\n")

# --- Create a synthetic test image ---
# We generate a simple image with Pillow rather than downloading one from the
# internet. This tests the vision pipeline without external dependencies.
# The image has colored shapes and text — enough for the model to describe.
def create_test_image(width, height, label="Test"):
    """Create a simple test image with shapes and text."""
    img = Image.new("RGB", (width, height), color=(240, 240, 255))
    draw = ImageDraw.Draw(img)

    # Draw colored rectangles
    draw.rectangle([20, 20, width // 3, height // 2], fill=(66, 133, 244))
    draw.rectangle([width // 3 + 10, 20, 2 * width // 3, height // 2], fill=(219, 68, 55))
    draw.rectangle([2 * width // 3 + 10, 20, width - 20, height // 2], fill=(244, 180, 0))

    # Draw a circle
    cx, cy = width // 2, 3 * height // 4
    r = min(width, height) // 6
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 157, 88))

    # Add text label
    draw.text((width // 2 - 30, height - 40), label, fill=(0, 0, 0))

    return img


print("Creating synthetic test image (640x480)...")
test_image = create_test_image(640, 480, label="Day 27 Lab")
test_image.save("/tmp/test_image.png")
print(f"Image created: {test_image.size[0]}x{test_image.size[1]} pixels\n")

# --- Process image with text query ---
# The processor converts the image into a sequence of visual tokens and
# combines them with the text tokens. The model then attends to both.
print("=" * 70)
print("EXPERIMENT 1: Image Description")
print("=" * 70)

question = "Describe what you see in this image in detail."
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": question},
        ],
    }
]

# Apply the chat template to format the multimodal input
text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text_input],
    images=[test_image],
    return_tensors="pt",
    padding=True,
)
inputs = inputs.to(model.device)

# Count image tokens — this shows the visual overhead
input_token_count = inputs["input_ids"].shape[1]
print(f"Total input tokens (image + text + template): {input_token_count}")

# Generate the description
start_time = time.perf_counter()
output_ids = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False,
)
gen_time = time.perf_counter() - start_time

# Decode only new tokens
new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
response = processor.decode(new_token_ids, skip_special_tokens=True)
image_tokens_generated = len(new_token_ids)
multimodal_tps = image_tokens_generated / gen_time

print(f"Tokens generated: {image_tokens_generated}")
print(f"Generation time: {gen_time:.1f}s ({multimodal_tps:.1f} tokens/sec)")
print(f"\nModel's description:\n  {response[:400]}\n")

# --- Text-only comparison ---
# Run a similar query without any image to measure the speed difference.
# This shows the computational overhead that visual processing adds.
print("=" * 70)
print("EXPERIMENT 2: Text-Only vs Multimodal Speed")
print("=" * 70)

text_only_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what a typical abstract painting with colored rectangles and circles looks like."},
        ],
    }
]

text_only_input = processor.apply_chat_template(text_only_messages, tokenize=False, add_generation_prompt=True)
text_only_inputs = processor(text=[text_only_input], return_tensors="pt", padding=True)
text_only_inputs = text_only_inputs.to(model.device)

text_only_token_count = text_only_inputs["input_ids"].shape[1]
print(f"Text-only input tokens: {text_only_token_count}")
print(f"Multimodal input tokens: {input_token_count}")
print(f"Image token overhead: {input_token_count - text_only_token_count} extra tokens\n")

start_time = time.perf_counter()
text_output_ids = model.generate(
    **text_only_inputs,
    max_new_tokens=200,
    do_sample=False,
)
text_gen_time = time.perf_counter() - start_time

text_new_tokens = text_output_ids[0][text_only_inputs["input_ids"].shape[1]:]
text_tps = len(text_new_tokens) / text_gen_time

print(f"Text-only: {len(text_new_tokens)} tokens in {text_gen_time:.1f}s ({text_tps:.1f} tok/s)")
print(f"Multimodal: {image_tokens_generated} tokens in {gen_time:.1f}s ({multimodal_tps:.1f} tok/s)")
print(f"Speed ratio: {text_tps / max(multimodal_tps, 0.01):.2f}x (text-only is faster)\n")

# --- Test with multiple image resolutions ---
# Image resolution directly affects how many visual tokens the processor
# creates. More pixels → more tokens → slower processing.
print("=" * 70)
print("EXPERIMENT 3: Resolution vs Token Count")
print("=" * 70)
print("Testing how image resolution affects visual token count...\n")

resolutions = [(320, 240), (640, 480), (1024, 768)]
resolution_results = []

for width, height in resolutions:
    img = create_test_image(width, height, label=f"{width}x{height}")

    res_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What colors do you see?"},
            ],
        }
    ]

    res_text = processor.apply_chat_template(res_messages, tokenize=False, add_generation_prompt=True)
    res_inputs = processor(text=[res_text], images=[img], return_tensors="pt", padding=True)
    res_inputs = res_inputs.to(model.device)

    res_token_count = res_inputs["input_ids"].shape[1]

    start_time = time.perf_counter()
    res_output = model.generate(**res_inputs, max_new_tokens=50, do_sample=False)
    res_time = time.perf_counter() - start_time

    res_new_tokens = len(res_output[0]) - res_inputs["input_ids"].shape[1]
    res_tps = res_new_tokens / res_time

    resolution_results.append((f"{width}x{height}", res_token_count, res_time, res_tps))

print(f"{'Resolution':<15} {'Input Tokens':<15} {'Gen Time (s)':<15} {'Tokens/sec':<12}")
print("-" * 57)
for res, tokens, t, tps in resolution_results:
    print(f"{res:<15} {tokens:<15} {t:<15.1f} {tps:<12.1f}")

# --- Summary metrics ---
print("\n" + "=" * 70)
print("SUMMARY METRICS")
print("=" * 70)
image_overhead = input_token_count - text_only_token_count
print(f"{'Metric':<35} {'Value':<20}")
print("-" * 55)
print(f"{'Model':<35} {MODEL_NAME}")
print(f"{'Parameters':<35} {param_count / 1e9:.1f}B")
print(f"{'Image tokens (640x480)':<35} {image_overhead}")
print(f"{'Text-only speed':<35} {text_tps:.1f} tok/s")
print(f"{'Multimodal speed':<35} {multimodal_tps:.1f} tok/s")
print(f"{'Visual overhead ratio':<35} {text_tps / max(multimodal_tps, 0.01):.2f}x slower")
print("-" * 55)
print("\nKey insight: Images are expensive in token space. A single 640x480")
print("image typically adds hundreds of tokens to the input, increasing both")
print("memory usage and time-to-first-token. Choose resolution carefully.")
