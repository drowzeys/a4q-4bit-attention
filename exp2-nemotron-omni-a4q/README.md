# Exp2 — A4Q on Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 (node r2, GB10)

**Model:** `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (30B / 3B active, hybrid
Mamba-2 + self-attention + MoE, ~21.6 GB NVFP4, 262K native context).
**Config:** TP=1, `--kv-cache-dtype nvfp4`, FlashInfer backend, `gpu-memory-utilization
0.70`, `--max-model-len 262144`, `--reasoning-parser nemotron_v3`, thinking off. A/B =
`VLLM_NVFP4_A4Q` 1 vs 0, same server.

**Verdict:** The strongest A4Q win in this study — TTFT −19% to −39% (scaling with context),
KV pool +4%, decode neutral, zero quality regression. A4Q kept as the standing r2 config.

---

## 1. Benchmark (A/B, short + 60K)

Short = 358-tok prompt, `max_tokens 512`, temp 0. Long = 60,018-tok prompt (passkeys at
25%/75%), `max_tokens 256`. Decode = true tokens via streaming usage.

| Config | short C1 decode | short C8 agg | 60K C1 decode | 60K TTFT | KV pool |
|---|---|---|---|---|---|
| **A4Q ON** | 56.7 tok/s | 233.5 tok/s | 56.9 tok/s | **9.39 s** | **30.62M tok** (233.6× @131K) |
| A4Q OFF | 57.1 | 229.7 | 58.2 | 12.03 s | 29.43M tok (224.6×) |
| **Δ (A4Q)** | ≈neutral | +1.7% | ≈neutral | **−21.9%** | **+4.0%** |

Short-C1 TTFT 0.118 s both configs. Decode is bandwidth-bound (NemotronH is mostly Mamba
layers), so A4Q's win lands entirely in **prefill/TTFT** + a bigger KV pool.

## 2. Evaluation (quality — identical both configs)

| Check | A4Q ON | A4Q OFF |
|---|---|---|
| Math word problems (first 5) | 5 / 5 | 5 / 5 |
| Math word problems (10) | 9 / 10 | 9 / 10 |
| Passkey @ 25% depth (60K) | ✅ "3187" | ✅ "3187" |
| Passkey @ 75% depth (60K) | ✅ "9254" | ✅ "9254" |
| Coherence | ✅ | ✅ |

Both configs miss the *same* Q10 ("$96 bill + $4 tip") with the *same* answer (32) — model
behavior, not A4Q. No divergence attributable to the kernel.

## 3. Concurrency sweep (A4Q=1, ~200-tok prompt, max_tokens 384, temp 0.7)

| C | Aggregate tok/s | Mean per-req tok/s | Mean TTFT (s) | Preemption |
|---|---|---|---|---|
| 1 | 58.0 | 58.0 | 0.13 | none |
| 2 | 103.0 | 51.5 | 0.62 | none |
| 4 | 156.7 | 39.3 | 1.33 | none |
| 8 | 229.5 | 29.1 | 0.41 | none |
| 12 | 237.7 | 20.4 | 0.75 | none |
| 16 | 318.9 | 20.6 | 1.23 | none |
| 20 | **329.1** | 17.0 | 0.89 | none |

Aggregate scales ~5.7× to 329 tok/s at C20; zero preemption / zero waiting reqs the whole
sweep (peak GPU KV usage 1.1% — the 30M-token nvfp4 pool dwarfs the prompts). C12/C16 wobble
is a temp-0.7 batching artifact; the aggregate trend is monotonic. Raw:
[`conc_sweep_a4q1.json`](conc_sweep_a4q1.json).

## 4. Context sweep (A4Q=1, single-stream, passkey @ 75% depth, max_tokens 192, temp 0)

Model max_position_embeddings = 262144 → swept the full range to a 261,500 cap.

| Target | Prompt tokens | TTFT (s) | Decode tok/s | Passkey |
|---|---|---|---|---|
| 6K | 6,058 | 2.44 | 57.2 | ✅ |
| 12K | 12,232 | 2.05 | 56.6 | ✅ |
| 24K | 24,482 | 3.65 | 56.9 | ✅ |
| 48K | 49,080 | 7.48 | 58.7 | ✅ |
| 96K | 98,178 | 17.43 | 58.5 | ✅ |
| 128K | 130,910 | 25.80 | 57.0 | ✅ |
| 192K | 196,423 | 47.22 | — * | ✅ |
| 256K | 261,299 | 75.08 | 59.2 | ✅ |

\* 192K reply self-terminated at 6 tokens (just the passkey) → decode-rate meaningless
there; TTFT valid. Passkey retrieved at **100% of depths through the full 256K**. Decode is
flat (~57–59 tok/s), context-independent — confirms bandwidth-bound decode. Raw:
[`ctx_sweep_a4q1.json`](ctx_sweep_a4q1.json).

### A/B context curve — A4Q ON vs OFF

| Context | A4Q TTFT | Baseline TTFT | A4Q gain |
|---|---|---|---|
| 48K | 7.48 s | 9.28 s | **−19.4%** |
| 96K | 17.43 s | 24.41 s | **−28.6%** |
| 256K | 75.08 s | 123.69 s | **−39.3%** |

A4Q saves ~49 s of prefill at 256K. Decode identical within noise both configs (~55–59
tok/s); KV pool essentially unchanged at 262144 ctx (29.89M vs 29.97M). Raw:
[`ctx_sweep_a4q0.json`](ctx_sweep_a4q0.json).

---

## Reproduce

```bash
# Serve (A4Q on, 262K). Baseline = prefix VLLM_NVFP4_A4Q=0
MAX_MODEL_LEN=262144 nohup ~/a4q/serve-nemotron-a4q.sh > serve.log 2>&1 &

python3 bench_a4q.py short          # short benchmark
python3 bench_a4q.py math           # eval
python3 bench_a4q.py conc conc_sweep_a4q1.json
python3 bench_a4q.py ctx  ctx_sweep_a4q1.json 6 12 24 48 96 128 192 256
```

Same FlashInfer-fork setup gotcha as Exp1 (see repo root README / Exp1). Additionally the
nemotron_v3 parser places non-streaming answers in the `reasoning` field with `content:
null` — streaming deltas deliver text normally, so benches are unaffected.

## Files
- `bench_a4q.py` — harness (`short` / `math` / `conc` / `ctx` modes)
- `conc_sweep_a4q1.json` — concurrency sweep
- `ctx_sweep_a4q1.json` / `ctx_sweep_a4q0.json` — context sweep + A/B
- `exp2_followup_summary.json` — run metadata, KV pools, provenance
