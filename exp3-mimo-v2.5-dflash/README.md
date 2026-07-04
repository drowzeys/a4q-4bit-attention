# Exp3 — MiMo-V2.5-NVFP4 TP=2 + DFlash/MTP spec-decode (nodes r0+r3, GB10)

**Model:** `lukealonso/MiMo-V2.5-NVFP4` (310B MoE / 15B active, **MLA + DSA** attention,
MIXED_PRECISION = NVFP4 experts + MXFP8 attn/dense-MLP, 171 GB / 39 shards).
**Drafter tested:** `XiaomiMiMo/MiMo-V2.5-DFlash`.
**Config:** TP=2 across r0(.4 head)+r3(.3 worker), 128K ctx, fp8 KV, `gpu-memory-utilization
0.84`, `--enforce-eager`, `--max-num-seqs 32`, `--max-num-batched-tokens 4096`,
`--distributed-executor-backend mp --nnodes 2`. Image `aeon-vllm-ultimate 0.24.0`.

**Why A4Q does not apply:** MiMo uses **MLA** — the KV cache is a compressed latent far below
A4Q's payoff width, and decode is absorbed-MLA, not paged QKᵀ. This experiment is therefore
a **speculative-decoding** study, not an A4Q one.

**Verdict:** **Baseline (no spec-decode) is the production config.** DFlash is
architecturally blocked; MTP loads but is a net slowdown.

---

## 1. Benchmark (baseline vs spec-decode)

Short prompt, `max_tokens 512`, streaming usage.

| Config | C1 tok/s | C8 agg tok/s | KV pool (tok) | Accept rate | Coherent |
|---|---|---|---|---|---|
| **Baseline (no spec)** | **23.1** | **77.5** | 468–492K | n/a | ✅ |
| MTP (k=1) | 17.4 | 51.4 | 459–464K | ~2–4% | ✅ but net **slower** |
| DFlash (k=8) | — | — | — | — | ❌ **won't load** |

At ~2–4% draft acceptance, MTP's verify overhead costs more than it saves (C1 −25%, C8
−34%). Baseline wins.

## 2. Evaluation (baseline, coherent throughout)

| Check | Result |
|---|---|
| Math word problems (10) | **10 / 10** |
| Coherence to 128K | ✅ |
| Temp-0 divergence (MTP vs baseline greedy) | 2/3 bit-identical; 1 diverged mid-content but stayed coherent |

The single divergence is fp8-KV + spec-decode batch-composition nondeterminism, **not** a
correctness bug (the AEON-campaign temperature block-expand bug was specifically checked for
and not present).

## 3. Concurrency sweep (baseline, 256-tok prompt)

| C | Aggregate tok/s |
|---|---|
| 1 | 17.5 |
| 4 | 50.4 |
| 8 | 75.8 |
| 12 | 92.6 |
| 16 | 120.2 |
| 20 | **127.1** |

Scales ~7.3× aggregate at C20, coherent throughout. Raw:
[`mimo-exp3-conc-sweep.json`](mimo-exp3-conc-sweep.json).

## 4. Context sweep (baseline, prefill → 64-tok decode)

| Target | TTFT (s) | Prefill tok/s | Decode tok/s | Coherent |
|---|---|---|---|---|
| 6K | 2.4 | 2422 | 21.0 | ✅ |
| 32K | 13.1 | 2366 | 20.5 | ✅ |
| 64K | 25.0 | 2479 | 20.2 | ✅ |
| 128K | 78.5 | 1577 | 18.4 | ✅ |

Bandwidth-bound decode ~18–21 tok/s (matches the GB10 ceiling for a ~85 GB/node shard). Raw:
[`mimo-exp3-ctx-sweep.json`](mimo-exp3-ctx-sweep.json).

---

## DFlash — BLOCKED (drafter architecture mismatch, not a loader bug)

```
KeyError: 'layers.0.self_attn.attention_sink_bias'   (qwen3_dflash.py:570)
```

The aeon image ships a **generic `DFlashQwen3Attention`** drafter (built for AEON's own
Qwen3 DFlash), which lacks three features the XiaomiMiMo MiMo-V2.5-DFlash drafter requires:

- `attention_sink_bias` (the missing weight),
- `attention_value_scale = 0.612`,
- `partial_rotary_factor = 0.5`.

Force-loading past the sink-bias would silently mis-model attention (partial rotary also
unimplemented) → *wrong* drafts, unvalidatable. Requires a MiMo-specific DFlash drafter
class ported into vLLM — staged for a future effort, not fixable by config.

## MTP — loads + coherent but ~2–4% acceptance (net slowdown)

`model-mtp.safetensors` loads via the natively-registered `MiMoV2MTPModel` after the same
quant fixes as the main model, but acceptance tops out ~4% — the MiMo MTP proposer is not
effectively wired into this aeon-0.24 spec path (the image was built around AEON's DFlash,
not the standard MTP proposer).

## The 6-fix MIXED_PRECISION loader chain (the real work)

The aeon 0.24 image registers `MiMoV2*` but its loader was broken for this
NVFP4-experts + MXFP8-attn/MLP checkpoint. Fixes in [`patches/`](patches/) (bind-mounted
over the image):

1. **`mimo_v2` unknown to transformers** → add `configuration_mimo_v2.py` + `auto_map` to
   the snapshot `config.json`.
2. **`_shard_fp8_qkv_proj` size crash (4096 vs 16384)** → force `_try_load_fp8_qkv_proj` →
   `False` so `QKVParallelLinear.weight_loader` does Q/K/V-aware TP slicing (per-row scales,
   canonical `[Q|K|V]` layout). (`mimo_v2.py`)
3. **`KeyError: weight_scale_inv`** → MXFP8 layers register `weight_scale`, checkpoint
   serializes `weight_scale_inv` (uint8) → rename in loader. (`mimo_v2.py`, `mimo_v2_mtp.py`)
4. **MXFP8 attn/MLP left unquantized** → add `mxfp8_config` + `if quant_algo=="MXFP8"`
   dispatch to `ModelOptMixedPrecisionConfig`. (`modelopt.py`)
5. **`KeyError: merger.mlp.0.bias`** → tolerate missing Omni vision-merger params, text-only
   serve. (`mimo_v2_omni.py`)
6. **KEY UNLOCK — garbage despite loading** → top-level `MiMoV2OmniForCausalLM` (the
   `SupportsQuant` owner) had no `packed_modules_mapping`, so fused `gate_up_proj` couldn't
   resolve the checkpoint's separate `gate_proj`/`up_proj` quant entries → all dense MLP +
   shared experts **silently unquantized**. Adding `packed_modules_mapping` → fully
   coherent. (`mimo_v2_omni.py`)

## Note: `mp --nnodes 2` works on vLLM 0.24

`--distributed-executor-backend mp --nnodes 2` sharded correctly here (each node loaded only
its ~85 GB TP shard, no double-load OOM). **Ray was not required** — a correction to older
guidance that multi-node TP needs Ray (true only for pre-0.24 mp).

---

## Reproduce

```bash
# On r0 (head), then r3 (worker). 3rd arg = spec mode: none | mtp | dflash
bash ~/mimo-exp3-serve.sh 0 none
ssh 10.100.10.3 'bash ~/mimo-exp3-serve.sh 1 none'   # --headless
python3 mimo-exp3-bench.py           # benchmark + eval + sweeps
```

Patches under `patches/` are bind-mounted over the aeon image's vLLM package. See
[`patches/NOTICE.md`](patches/NOTICE.md) for provenance/licensing.

## Files
- `mimo-exp3-serve.sh` — TP=2 multi-node launcher (spec mode arg)
- `mimo-exp3-bench.py` — harness
- `mimo-exp3-baseline-temp0.json` — benchmark + eval + divergence check
- `mimo-exp3-conc-sweep.json` — concurrency sweep
- `mimo-exp3-ctx-sweep.json` — context sweep
- `patches/` — the 6-fix MIXED_PRECISION loader chain (+ `qwen3_dflash.py` showing the block)
