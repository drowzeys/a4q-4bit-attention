#!/usr/bin/env python3
"""Exp1 follow-up sweeps on qwen36-a4q @ 10.100.10.1:8000 (r1).
Usage:
  python3 bench_exp1_sweep.py conc <tag>                 # concurrency sweep C in {1,2,4,8,12,16,20}
  python3 bench_exp1_sweep.py ctx  <tag> <points-csv>    # context sweep, points in K-tokens e.g. 6,12,24
"""
import json, sys, time, threading, urllib.request, re

BASE = "http://10.100.10.1:8000/v1/chat/completions"
MODEL = "qwen36-a4q"

def stream_request(messages, max_tokens, temperature=0.0, timeout=3600):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE, data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None
    text = []; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                c = delta.get("content") or delta.get("reasoning_content") or ""
                if c:
                    if t_first is None:
                        t_first = time.time()
                    t_last = time.time()
                    text.append(c)
    ttft = (t_first - t0) if t_first else None
    ct = usage.get("completion_tokens") if usage else None
    pt = usage.get("prompt_tokens") if usage else None
    decode_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    tps = (ct - 1) / decode_s if (ct and decode_s) else None
    return {"ttft": ttft, "completion_tokens": ct, "prompt_tokens": pt,
            "decode_tps": tps, "wall": time.time() - t0, "text": "".join(text)}

SHORT_PROMPT = ("You are helping document a fictional research lab. Write a detailed, "
    "coherent description of the lab's daily operations. Context: The Meridian Institute "
    "sits on a windswept plateau in northern Iceland, where forty-two researchers study "
    "atmospheric electricity and auroral plasma dynamics. Every morning the instrument "
    "teams calibrate their magnetometers before dawn, because the solar-quiet interval "
    "gives the cleanest baseline readings. The institute operates three field stations "
    "connected by fiber-optic links buried beneath the permafrost, and each station logs "
    "roughly nine terabytes of ionospheric radar data per week. Funding comes from a "
    "consortium of five universities, and the annual review each September determines "
    "which experiments continue through the polar winter. The cafeteria serves fish "
    "stew on Thursdays, a tradition dating back to the founding director, who believed "
    "shared meals kept morale high during the long darkness. Describe a typical Tuesday "
    "from 6am to midnight, including instrument work, data review meetings, and the "
    "evening aurora watch rotation.")

def conc_point(C):
    results = [None] * C
    errors = [None] * C
    def worker(i):
        try:
            results[i] = stream_request(
                [{"role": "user", "content": SHORT_PROMPT + f" (variant {i})"}],
                384, temperature=0.7)
        except Exception as e:
            errors[i] = str(e)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    good = [r for r in results if r]
    total_ct = sum(r["completion_tokens"] or 0 for r in good)
    ttfts = [r["ttft"] for r in good if r["ttft"]]
    first = min(ttfts) if ttfts else 0
    agg = total_ct / (wall - first) if wall > first else None
    mean_dec = sum(r["decode_tps"] for r in good if r["decode_tps"]) / max(1, len(good))
    return {"C": C, "agg_tps": round(agg, 2) if agg else None,
            "mean_req_decode_tps": round(mean_dec, 2),
            "mean_ttft": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
            "total_tokens": total_ct, "wall": round(wall, 2),
            "n_errors": sum(1 for e in errors if e),
            "errors": [e for e in errors if e][:3]}

FILLER = ("The grain harvest in the valley proceeded on schedule that season, and the "
    "millers recorded each delivery in heavy canvas ledgers. Wagons arrived from the "
    "eastern farms before noon, their axles creaking under sacks of barley and rye. "
    "The weighing house stood beside the river, where the current turned the great "
    "stone wheels day and night. Children gathered near the loading dock to watch the "
    "teamsters stack the sacks in tidy pyramids. ")
TOK_PER_REPEAT = 88.06  # calibrated: 61643 tok / 700 repeats

def build_ctx_prompt(target_k):
    target = target_k * 1000
    n = max(4, int((target - 60) / TOK_PER_REPEAT))
    at = int(n * 0.75)
    parts = []
    for i in range(n):
        if i == at:
            parts.append("The passkey is 7429. ")
        parts.append(FILLER)
    parts.append("\n\nWhat is the passkey? Reply with just the number.")
    return "".join(parts)

def ctx_point(target_k):
    prompt = build_ctx_prompt(target_k)
    try:
        r = stream_request([{"role": "user", "content": prompt}], 192, temperature=0.0)
    except Exception as e:
        return {"target_k": target_k, "error": str(e)[:300]}
    return {"target_k": target_k, "prompt_tokens": r["prompt_tokens"],
            "ttft": round(r["ttft"], 2) if r["ttft"] else None,
            "decode_tps": round(r["decode_tps"], 2) if r["decode_tps"] else None,
            "completion_tokens": r["completion_tokens"],
            "passkey_ok": "7429" in r["text"],
            "tail": r["text"][-120:]}

def main():
    mode, tag = sys.argv[1], sys.argv[2]
    out = {"mode": mode, "tag": tag, "points": []}
    if mode == "conc":
        for C in [1, 2, 4, 8, 12, 16, 20]:
            print(f"=== conc C={C} ===", flush=True)
            p = conc_point(C)
            out["points"].append(p)
            print(json.dumps(p), flush=True)
            time.sleep(3)
    elif mode == "ctx":
        pts = [int(x) for x in sys.argv[3].split(",")]
        for k in pts:
            print(f"=== ctx {k}K ===", flush=True)
            p = ctx_point(k)
            out["points"].append(p)
            print(json.dumps(p), flush=True)
            time.sleep(3)
    with open(f"/home/keyspark/a4q-lab/results_exp1_{mode}_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved results_exp1_{mode}_{tag}.json", flush=True)

if __name__ == "__main__":
    main()
