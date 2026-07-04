#!/usr/bin/env python3
"""Exp1 bench: A4Q vs baseline on qwen36-a4q @ 10.100.10.1:8000 (r1).
Measures TRUE decode tok/s via streaming (completion_tokens / wall-after-first-token).
Usage: python3 bench_exp1.py <tag>   (tag e.g. a4q1 / a4q0)
"""
import json, sys, time, threading, urllib.request, re

BASE = "http://10.100.10.1:8000/v1/chat/completions"
MODEL = "qwen36-a4q"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"

def stream_request(messages, max_tokens, temperature=0.0, timeout=1800):
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

def bench_short_c1():
    return stream_request([{"role": "user", "content": SHORT_PROMPT}], 512, temperature=0.7)

def bench_short_c8():
    results = [None] * 8
    def worker(i):
        results[i] = stream_request([{"role": "user", "content": SHORT_PROMPT + f" (variant {i})"}], 512, temperature=0.7)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    total_ct = sum(r["completion_tokens"] or 0 for r in results)
    first_token = min(r["ttft"] for r in results if r["ttft"])
    agg = total_ct / (wall - first_token)
    return {"agg_tps": agg, "total_tokens": total_ct, "wall": wall,
            "mean_ttft": sum(r["ttft"] for r in results if r["ttft"]) / 8,
            "per_req_tps": [round(r["decode_tps"], 1) if r["decode_tps"] else None for r in results]}

FILLER = ("The grain harvest in the valley proceeded on schedule that season, and the "
    "millers recorded each delivery in heavy canvas ledgers. Wagons arrived from the "
    "eastern farms before noon, their axles creaking under sacks of barley and rye. "
    "The weighing house stood beside the river, where the current turned the great "
    "stone wheels day and night. Children gathered near the loading dock to watch the "
    "teamsters stack the sacks in tidy pyramids. ")

def build_long_prompt(n_repeat=700):
    quarter = n_repeat // 4
    parts = []
    for i in range(n_repeat):
        if i == quarter:
            parts.append("The alpha passkey is 7429. ")
        if i == 3 * quarter:
            parts.append("The omega passkey is 3186. ")
        parts.append(FILLER)
    return "".join(parts)

LONG = build_long_prompt()

def bench_long():
    perf = stream_request([{"role": "user", "content": LONG +
        "\n\nSummarize the passage above in two sentences."}], 256)
    pk1 = stream_request([{"role": "user", "content": LONG +
        "\n\nWhat is the alpha passkey? Reply with just the number."}], 512)
    pk2 = stream_request([{"role": "user", "content": LONG +
        "\n\nWhat is the omega passkey? Reply with just the number."}], 512)
    return perf, pk1, pk2

MATH = [
    ("A train travels 60 miles in 1.5 hours. What is its average speed in mph?", "40"),
    ("If 3x + 7 = 25, what is x?", "6"),
    ("Sarah has 3 times as many apples as Tom. Together they have 48 apples. How many apples does Sarah have?", "36"),
    ("What is 17 multiplied by 23?", "391"),
    ("A shirt costs $80 after a 20% discount. What was the original price in dollars?", "100"),
    ("What is the sum of the first 20 positive integers?", "210"),
    ("A rectangle has a perimeter of 36 and a length of 10. What is its area?", "80"),
    ("A car worth $20,000 depreciates by 10% per year. What is it worth in dollars after exactly 2 years?", "16200"),
    ("Compute 2^10 - 3^5.", "781"),
    ("Alice is twice as old as Bob. In 6 years the sum of their ages will be 48. How old is Bob now?", "12"),
]

def bench_math():
    score = 0; details = []
    for q, gold in MATH:
        r = stream_request([{"role": "user", "content": q +
            " Think briefly, then give your final answer on the last line in the exact format ANSWER: <number>"}],
            2048, temperature=0.0)
        m = re.findall(r"ANSWER:\s*\$?([\d,\.]+)", r["text"])
        got = m[-1].replace(",", "").rstrip(".") if m else None
        if got and got.endswith(".0"): got = got[:-2]
        ok = (got == gold)
        score += ok
        details.append({"q": q[:40], "gold": gold, "got": got, "ok": ok})
    return score, details

def main():
    out = {"tag": TAG}
    print(f"=== {TAG}: short C1 ===", flush=True)
    r = bench_short_c1()
    out["short_c1"] = {k: v for k, v in r.items() if k != "text"}
    out["short_c1_sample"] = r["text"][:300]
    print(json.dumps(out["short_c1"]), flush=True)
    print(f"=== {TAG}: short C8 ===", flush=True)
    out["short_c8"] = bench_short_c8()
    print(json.dumps(out["short_c8"]), flush=True)
    print(f"=== {TAG}: long 60K ===", flush=True)
    perf, pk1, pk2 = bench_long()
    out["long_perf"] = {k: v for k, v in perf.items() if k != "text"}
    out["long_perf_sample"] = perf["text"][:300]
    out["pk_alpha"] = {"ttft": pk1["ttft"], "text": pk1["text"][-200:], "ok": "7429" in pk1["text"]}
    out["pk_omega"] = {"ttft": pk2["ttft"], "text": pk2["text"][-200:], "ok": "3186" in pk2["text"]}
    print(json.dumps({"long_perf": out["long_perf"], "pk_alpha_ok": out["pk_alpha"]["ok"],
                      "pk_omega_ok": out["pk_omega"]["ok"]}), flush=True)
    print(f"=== {TAG}: math 10 ===", flush=True)
    score, details = bench_math()
    out["math_score"] = score
    out["math_details"] = details
    print(json.dumps({"math": f"{score}/10", "details": details}), flush=True)
    with open(f"/home/keyspark/a4q-lab/results_exp1_{TAG}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved", f"results_exp1_{TAG}.json", flush=True)

if __name__ == "__main__":
    main()
