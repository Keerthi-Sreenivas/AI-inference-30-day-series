# 30 Days of AI Inference Engineering

A hands-on, 90-second-video learning series that takes you from zero to deploying production inference — one concept per day, one prompt per day.

## How This Works

Each day covers one concept in AI inference engineering:

1. Watch the 90-second video where I introduce the concept
2. Clone this repo and `cd` into the day's folder
3. Run the prompt in your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold the project
4. Explore the generated code, tweak it, break it, learn from it

## Prerequisites

- Python 3.10+
- 16GB RAM minimum (32GB recommended)
- GPU optional for Week 1 (required from Week 2+)
- A coding assistant (Kiro CLI recommended)

## Curriculum

### Week 1: Foundations & First Inference
| Day | Concept | Tools |
|-----|---------|-------|
| 1 | Your First Local Inference with Ollama | Ollama |
| 2 | Tokens & Tokenizers — How Text Becomes Numbers | HF Tokenizers |
| 3 | Loading Models with Hugging Face Transformers | Transformers |
| 4 | Sampling Strategies — Temperature, Top-K, Top-P | Transformers |
| 5 | Model Formats & Precision — FP16, BF16, INT8, INT4 | Transformers, safetensors |
| 6 | GGUF & llama.cpp — Running Models on CPU | llama.cpp |
| 7 | Understanding config.json — What Every Parameter Means | HF Hub |

### Week 2: Quantization & Optimization
| Day | Concept | Tools |
|-----|---------|-------|
| 8 | Quantization Theory — Why Fewer Bits Still Work | bitsandbytes |
| 9 | GGUF Quantization — Convert Your Own Model | llama.cpp |
| 10 | 4-bit Inference with bitsandbytes | Transformers, bitsandbytes |
| 11 | AWQ & GPTQ — GPU Quantization Formats | AutoAWQ, AutoGPTQ |
| 12 | The KV Cache — Why Inference Eats Memory | Transformers |
| 13 | Flash Attention — Faster, Leaner Attention | flash-attn |
| 14 | Benchmarking Inference — Tokens/sec, Latency, VRAM | Custom scripts |

### Week 3: Serving & Production
| Day | Concept | Tools |
|-----|---------|-------|
| 15 | vLLM — Production Inference Serving | vLLM |
| 16 | PagedAttention & Continuous Batching | vLLM |
| 17 | SGLang — RadixAttention and Serving Comparison | SGLang, vLLM |
| 18 | Multi-GPU Inference — Tensor Parallelism and Ray | vLLM, SGLang, Ray Serve |
| 19 | Structured Output — JSON Mode & Constrained Decoding | vLLM, SGLang, Outlines |
| 20 | Building an Engine-Agnostic Inference API | FastAPI, vLLM/SGLang/NIM |
| 21 | Load Testing Across Engines | Locust, vLLM, SGLang, NIM |

### Week 4: Advanced Topics
| Day | Concept | Tools |
|-----|---------|-------|
| 22 | Speculative Decoding — Draft Models for Speed | vLLM, SGLang |
| 23 | ONNX Runtime — Cross-Platform Optimization | Optimum, ONNX |
| 24 | TensorRT-LLM, Triton & NVIDIA NIMs | TensorRT-LLM, Triton, NIM |
| 25 | LoRA & QLoRA — Lightweight Fine-Tuning | PEFT, bitsandbytes |
| 26 | RAG Inference Pipeline — Retrieval + Generation | Transformers, FAISS |
| 27 | Multimodal Inference — Vision-Language Models | Transformers |
| 28 | Monitoring & Observability for Inference | Prometheus |
| 29 | Cost Optimization — Cloud vs Local, Engine Comparison | vLLM, SGLang, NIM, Ray |
| 30 | **Capstone** — Production Inference Service from Scratch | vLLM, SGLang, NIM, FastAPI, Docker, Prometheus |

## Repository Structure

```
day01-first-local-inference/
├── README.md          # Concept explanation + video link
├── prompt.md          # The prompt to run in your coding assistant
├── requirements.txt   # Python dependencies
└── lab/main.py        # The generated/reference implementation
```

## Quick Start

```bash
git clone https://github.com/Keerthi-Sreenivas/AI-inference-30-day-series.git
cd AI-inference-30-day-series/day01-first-local-inference
# Read the README, then paste prompt.md into your coding assistant
```

## Resources

- [vLLM Docs](https://docs.vllm.ai/)
- [SGLang](https://github.com/sgl-project/sglang)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [NVIDIA NIM](https://developer.nvidia.com/nim)
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)

## License

MIT
