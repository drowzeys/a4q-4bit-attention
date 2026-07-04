# Exp1 — A4Q on Qwen3.6-27B-NVFP4 (node r1, GB10)

**Model:** `nvidia/Qwen3.6-27B-NVFP4` (27B, hybrid Gated-DeltaNet + gated attention, ~20.4 GB
NVFP4, 262K native context).
**Config:** TP=1, `--kv-cache-dtype nvfp4`, FlashInfer backend, `gpu-memory-utilization
0.70`, `--max-model-len 262144`. A/B = `VLLM_NVFP4_A4Q` 1 vs 0, same server.

**Verdict:** A4Q is a strict win — TTFT −6% to −23% (scaling with context), decode +4%, KV
pool +1.4%, zero quality regression.

---

## 1. Benchmark (A/B, short + 60K)

Short = 205-tok prompt, `max_tokens 512`, temp 0.7. Long = 61,643-tok prompt, `max_tokens
256`, temp 0. Decode = true tokens via streaming usage.

| Config | short C1 decode | short C8 agg | 60K C1 decode | 60K TTFT | short TTFT | KV pool |
|---|---|---|---|---|---|---|
| **A4Q ON** | **11.65 tok/s** | **82.3 tok/s** | **10.91 tok/s** | **75.4 s** | **0.27 s** | 58.36 GiB / 3.14M tok |
| A4Q OFF | 11.20 | 80.8 | 10.51 | 84.8 s | 0.66 s | 57.56 GiB / 3.09M tok |
| **Δ (A4Q)** | +4.1% | +1.9% | +3.9% | **−11.1%** | **−59%** | +1.4% |

## 2. Evaluation (quality — identical both configs)

| Check | A4Q ON | A4Q OFF |
|---|---|---|
| Math word problems (10) | **10 / 10** | 10 / 10 |
| Passkey @ 25% depth (60K) | ✅ "7429" | ✅ "7429" |
| Passkey @ 75% depth (60K) | ✅ "3186" | ✅ "3186" |
| Coherence | ✅ coherent English | ✅ coherent English |

No divergence between configs — A4Q's fp4 QK MMA preserves outputs bit-for-bit at the
answer level.

## 3. Concurrency sweep (A4Q=1, ~200-tok prompt, max_tokens 384, temp 0.7)

| C | Aggregate tok/s | Mean per-req tok/s | Mean TTFT (s) | Errors | Preemption |
|---|---|---|---|---|---|
| 1 | 11.32 | 11.29 | 1.73 | 0 | none |
| 2 | 21.73 | 10.84 | 0.89 | 0 | none |
| 4 | 42.17 | 10.56 | 1.24 | 0 | none |
| 8 | 79.98 | 9.97 | 2.02 | 0 | none |
| 12 | 110.99 | 9.38 | 2.63 | 0 | none |
| 16 | 137.00 | 8.81 | 2.90 | 0 | none |
| 20 | **153.31** | 7.97 | 3.61 | 0 | none |

Near-linear aggregate scaling to 13.5× at C20; per-request degrades gracefully (bandwidth
shared). Zero preemption/eviction the whole sweep — the 3.14M-token KV pool is nowhere near
pressure at these ~600-token working sets, so C20 is the top of the requested range, not a
ceiling. Raw: [`results_exp1_conc_a4q1.json`](results_exp1_conc_a4q1.json).

## 4. Context sweep (A4Q=1, single-stream, passkey @ 75% depth, max_tokens 192, temp 0)

| Target | Prompt tokens | TTFT (s) | Decode tok/s | Passkey |
|---|---|---|---|---|
| 6K | 5,929 | 5.70 | 11.34 | ✅ |
| 12K | 11,913 | 11.72 | 11.24 | ✅ |
| 24K | 23,881 | 25.30 | 10.92 | ✅ |
| 48K | 47,905 | 56.56 | 10.67 | ✅ |
| 96K | 95,865 | 130.65 | 10.22 | ✅ |
| 128K | 127,809 | 192.66 | 9.98 | ✅ |
| 192K | 191,785 | 341.09 | 9.39 | ✅ |
| 256K | 255,761 | 523.11 | 8.81 | ✅ |

262K opened cleanly at util 0.70 (KV pool stayed positive, no fallback). Passkey correct at
**every** depth through 256K. Decode holds 11.3→8.8 tok/s (bandwidth-bound, mild KV-growth
falloff); TTFT super-linear (compute-bound prefill). Raw:
[`results_exp1_ctx_a4q1.json`](results_exp1_ctx_a4q1.json).

### A/B context curve — A4Q ON vs OFF

| Target | A4Q TTFT | Baseline TTFT | A4Q gain | Decode (A4Q/base) | Passkey |
|---|---|---|---|---|---|
| 48K | 56.56 s | 60.34 s | **−6.3%** | 10.67 / 10.69 | ✅ both |
| 96K | 130.65 s | 153.13 s | **−14.7%** | 10.22 / 10.11 | ✅ both |
| 256K | 523.11 s | 678.07 s | **−22.9%** | 8.81 / 8.81 | ✅ both |

A4Q's prefill advantage widens with context — the signature of accelerating O(N²) attention
prefill. At 256K it cuts single-stream prefill from ~11 min to ~8.7 min. Raw:
[`results_exp1_ctx_a4q0.json`](results_exp1_ctx_a4q0.json).

---

## Reproduce

```bash
# Serve (A4Q on). Baseline = prefix VLLM_NVFP4_A4Q=0
sed -i "s/--max-model-len 131072/--max-model-len 262144/" ~/a4q/serve-qwen-a4q.sh
nohup ~/a4q/serve-qwen-a4q.sh > serve.log 2>&1 &

# Benches
python3 bench_exp1.py a4q1            # short + 60K benchmark + eval
python3 bench_exp1_sweep.py conc a4q1                 # concurrency
python3 bench_exp1_sweep.py ctx  a4q1 6,12,24,48,96,128,192,256   # context
python3 bench_exp1_sweep.py ctx  a4q0 48,96,256                   # A/B baseline
```

**Setup gotcha:** the A4Q venv must actually install the jethac FlashInfer fork —
`--system-site-packages` silently leaves stock flashinfer 0.6.12 in place (→ `plan() got an
unexpected keyword argument 'use_nvf4_qk'` at warmup). Fix: `pip install --force-reinstall
--no-deps <fork>.whl` + `pip install flashinfer-cubin==0.6.13` + `chown` any root-owned
`~/.cache/flashinfer` + `FLASHINFER_DISABLE_VERSION_CHECK=1`.

## Files
- `bench_exp1.py`, `bench_exp1_sweep.py` — harnesses
- `results_exp1_a4q1.json` / `results_exp1_a4q0.json` — benchmark + eval
- `results_exp1_conc_a4q1.json` — concurrency sweep
- `results_exp1_ctx_a4q1.json` / `results_exp1_ctx_a4q0.json` — context sweep + A/B
