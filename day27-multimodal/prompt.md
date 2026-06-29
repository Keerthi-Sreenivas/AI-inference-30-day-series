# Day 27 Prompt — Multimodal Inference

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers torch accelerate qwen-vl-utils Pillow
```

## Step 2: Model download

The script uses `Qwen/Qwen2.5-VL-7B-Instruct` — a vision-language model that downloads automatically on first run (~15 GB). It's Apache 2.0 licensed and ungated (no account needed).

## Step 3: Create the project

Create a `requirements.txt` with:

```
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
qwen-vl-utils>=0.0.2
Pillow>=10.0.0
```

Create a Python script called `main.py` that performs multimodal inference: processing both images and text through a vision-language model. Use `transformers` with `Qwen/Qwen2.5-VL-7B-Instruct` to demonstrate how the same model architecture handles visual and textual inputs.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Load the vision-language model** — Load `Qwen/Qwen2.5-VL-7B-Instruct` with its processor using `AutoProcessor` and `AutoModelForCausalLM` (or the VL-specific class). Use `device_map="auto"`. Print model parameter count.

2. **Create a synthetic test image** — Generate a simple test image using Pillow (e.g., a colored rectangle with text drawn on it, or geometric shapes). This avoids external image downloads while still testing the vision pipeline. Save it and print its dimensions.

3. **Process image with text query** — Send the test image with a question (e.g., "Describe what you see in this image") through the model's processor and generate a response. Measure and print the number of tokens generated.

4. **Count image tokens** — Inspect the processor output to determine how many tokens the image was converted to. Print this count — it shows the visual token overhead compared to text-only inference.

5. **Compare text-only vs multimodal** — Run the same question without an image (text-only mode). Compare generation speed (tokens/sec) between the two modes to show the computational overhead of processing visual inputs.

6. **Test with multiple image sizes** — Process images at 2-3 different resolutions and show how the number of image tokens changes. Print a comparison table of resolution → token count → processing time.

7. **Print summary metrics** — Print: total image tokens processed, text-only vs multimodal tokens/sec, and the visual token overhead ratio.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — load model, process image, compare modes
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block (for model loading)

### Verification

After running the script, you should see:

- Model loaded with parameter count displayed
- A synthetic image created and its dimensions printed
- The model's description of the image content
- Image token count (typically 100-1000+ depending on resolution)
- Speed comparison between text-only and multimodal inference
- A table showing how image resolution affects token count and speed
