# A4Q 4-bit Application (by Jetha Chan) — Optimizing Attention for GQA / MoE / Hybrid-Attention LLMs

Benchmarks, evaluations, concurrency + context sweeps, and reproduction artifacts for
**A4Q** — native NVFP4 (4-bit) block-scaled QK attention — measured on a **4× NVIDIA DGX
Spark (GB10, sm_121a)** cluster.

> **A4Q is the work of [Jetha Chan (@jetha)](https://x.com/jetha)** — see
> [jethac/flashinfer](https://github.com/jethac/flashinfer) (the kernel) and
> [jethac/blackwell-isa-probe](https://github.com/jethac/blackwell-isa-probe) (the ISA
> ground-truth). This repository contains **independent third-party measurements** of A4Q
> on DGX Spark hardware, not the kernel itself. All credit for A4Q belongs to the author.

## What A4Q does

Consumer Blackwell (RTX 50, DGX Spark / GB10) has **real hardware** dense NVFP4
`mma.sync kind::mxf4nvf4` block-scaled tensor-core instructions — proven at the ISA level
(GB10 is instruction-identical to the RTX 5090). The shipping fp4 attention path ignored
this and unpacked 4-bit KV through a ~9-instruction fp16 chain, making fp4 attention
*slower* than bf16. **A4Q replaces that with a native fp4 QKᵀ MMA** (query quantized to
e2m1 on the fly, KV kept fp4 in cache, fp8 scale per 16 values), engaged by
`VLLM_NVFP4_A4Q=1`.

## Headline result

**A4Q accelerates the attention portion of prefill, so its Time-To-First-Token (TTFT)
advantage grows with context length — at zero decode cost and zero quality change.**

| Model | 48K TTFT | 96K TTFT | 256K TTFT | Decode | Quality |
|---|---|---|---|---|---|
| **Nemotron-3-Omni-30B** | −19% | −29% | **−39%** | neutral | identical |
| **Qwen3.6-27B** | −6% | −15% | **−23%** | +4% | identical |

Both models are Mamba/attention hybrids; Nemotron gains ~1.7× what Qwen does because it has
proportionally more attention layers for A4Q to accelerate. Passkey retrieval was 100% at
every depth through the full **256K** context in both A4Q-on and A4Q-off configs.

## Experiments in this repo

| Dir | Model | Attention | A4Q applies? | Result |
|---|---|---|---|---|
| [`exp1-qwen3.6-a4q/`](exp1-qwen3.6-a4q/) | Qwen3.6-27B-NVFP4 | GDN + gated attn (hybrid) | ✅ | Win — TTFT −6…−23%, decode +4% |
| [`exp2-nemotron-omni-a4q/`](exp2-nemotron-omni-a4q/) | Nemotron-3-Nano-Omni-30B-A3B-NVFP4 | Mamba-2 + attn + MoE (hybrid) | ✅ | Win — TTFT −19…−39%, KV +4% |

Each experiment directory contains its own README with **benchmark, evaluation,
concurrency sweep, and context sweep** tables, plus the raw JSON results, bench harness,
and serve scripts.

## Scope: where A4Q applies

A4Q targets **paged GQA / hybrid attention** where the KV cache holds per-head keys/values.
Models with **MLA** attention (DeepSeek-V4, GLM-5.2, MiMo-V2.5) keep a compressed latent in
cache — far below A4Q's payoff width, with decode running absorbed-MLA kernels rather than
paged QKᵀ — so A4Q does not apply to them.

## Test platform

- **4× NVIDIA DGX Spark** (GB10 Grace-Blackwell, sm_121a, 128 GB unified memory each), 200G
  ConnectX-7 RoCE fabric.
- vLLM: jethac A4Q fork wheel `0.1.dev1+g884e01b4f.sm121a.arm64.a4q` (exp1/2);
  `aeon-vllm-ultimate 0.24.0` (exp3). FlashInfer 0.6.13 (jethac fork) + matched cubins.
- CUDA 13.0, Python 3.12, aarch64.

## Method notes (apply to all experiments)

- **True decode tok/s**: streaming with `stream_options {"include_usage": true}`,
  computed as `(completion_tokens − 1) / (t_last − t_first_token)` — never wall-clock or
  chunk-count.
- **A/B isolation**: same server, same node, only `VLLM_NVFP4_A4Q` flipped 1↔0. Engagement
  verified per-run by the log marker `A4Q: nvf4 block-scaled QK MMA enabled for FA2 NVFP4
  prefill.` (present iff A4Q=1).
- `gpu-memory-utilization` held at 0.70 (exp1/2) / 0.84 (exp3) throughout — never tuned
  between A/B legs.
- Passkey retrieval: a unique key hidden at a fixed depth in a long padded prompt, then
  requested — verifies long-range attention correctness, not just perplexity.

## Credits & license

- **A4Q kernel, FlashInfer fork, and Blackwell ISA probe**: Jetha Chan (@jetha) — MIT.
- Measurement data, harnesses, and documentation in this repo: MIT (see [`LICENSE`](LICENSE)).
