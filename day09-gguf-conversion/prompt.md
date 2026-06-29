# Day 9 Prompt — GGUF Conversion

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install huggingface-hub transformers torch numpy sentencepiece
```

## Step 2: Clone and build llama.cpp

You need the llama.cpp repository for its conversion script and quantize binary:

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Install the conversion script's Python dependencies
pip install -r requirements/requirements-convert_hf_to_gguf.txt

# Build the quantize tool
cmake -B build
cmake --build build --config Release --target llama-quantize
cd ..
```

**macOS:** Requires Xcode Command Line Tools (`xcode-select --install`) and cmake (`brew install cmake`).

**Linux:** Requires `build-essential` and `cmake` (`sudo apt install build-essential cmake`).

**Windows:** Use WSL and follow Linux instructions, or install Visual Studio Build Tools + cmake.

## Step 3: Create the project

Create a `requirements.txt` with:

```
huggingface-hub>=0.24.0
transformers>=5.0.0
torch>=2.4.0
numpy>=1.24.0
sentencepiece>=0.2.0
```

Create a Python script called `main.py` that downloads `HuggingFaceTB/SmolLM2-1.7B` (ungated, Apache 2.0), converts it to GGUF format at multiple quantization levels using the llama.cpp tools, and prints the output file size in megabytes for each level.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Download the model** — Use `huggingface_hub.snapshot_download()` to fetch `HuggingFaceTB/SmolLM2-1.7B` to a local directory. Print the download path and total size of the downloaded model files in MB.

2. **Verify llama.cpp is available** — Check that the `llama.cpp/convert_hf_to_gguf.py` script and the `llama.cpp/build/bin/llama-quantize` binary exist. If either is missing, print a clear error message explaining how to clone and build llama.cpp, then exit gracefully.

3. **Convert to FP16 GGUF** — Run `convert_hf_to_gguf.py` via `subprocess` to produce an FP16 baseline GGUF file. Print the output file size in MB.

4. **Quantize to Q8_0** — Run `llama-quantize` to produce a Q8_0 quantized GGUF from the FP16 baseline. Print the output file size in MB.

5. **Quantize to Q4_K_M** — Run `llama-quantize` to produce a Q4_K_M quantized GGUF from the FP16 baseline. Print the output file size in MB.

6. **Quantize to Q2_K** — Run `llama-quantize` to produce a Q2_K quantized GGUF from the FP16 baseline. Print the output file size in MB.

7. **Print summary table** — At the end, print a formatted comparison table showing each quantization level with its file size in MB and the compression ratio relative to the FP16 baseline. This is the key quantitative output.

### Code Style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — download, verify, convert, quantize, summarize
- Educational inline comments — at least one comment per logical block
- At most one `try/except` for subprocess calls
- Use `subprocess.run()` for calling llama.cpp tools
- Use `os.path.getsize()` to measure output files

### Verification

After running the script, you should see:

- The download path of the SmolLM2-1.7B model
- Total source model size printed in MB
- An FP16 GGUF file created with its size printed (~3.4 GB)
- A Q8_0 GGUF file with its size (~1.8 GB)
- A Q4_K_M GGUF file with its size (~1.0 GB)
- A Q2_K GGUF file with its size (~0.7 GB)
- A summary table comparing all levels with compression ratios
- Total script runtime: 5–15 minutes depending on hardware (excluding initial model download)
