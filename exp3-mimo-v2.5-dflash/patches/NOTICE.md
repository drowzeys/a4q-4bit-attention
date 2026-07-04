# Provenance & licensing — MiMo loader patches

These files are **modified copies of vLLM source** (Apache License 2.0), as shipped in the
`ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-01-v0.24.0` image, patched to load the
`lukealonso/MiMo-V2.5-NVFP4` MIXED_PRECISION checkpoint (NVFP4 experts + MXFP8 attn/MLP) on
DGX Spark (GB10 / sm_121a).

- `mimo_v2.py`, `mimo_v2_mtp.py`, `mimo_v2_omni.py` — MiMo model + MTP + Omni wrappers
  (QKV weight loading, weight_scale_inv rename, packed_modules_mapping).
- `modelopt.py` — ModelOpt quant config (added MXFP8 sub-config + dispatch).
- `qwen3_dflash.py` — **unmodified**, included only to show where the DFlash drafter
  architecture mismatch raises (`attention_sink_bias`, line ~570).

Original vLLM code: https://github.com/vllm-project/vllm (Apache-2.0). Modifications are
provided under the same Apache-2.0 terms for reproducibility. No warranty.
