#!/usr/bin/env python3
"""Exp2 A4Q bench: TRUE decode tok/s via streaming + include_usage."""
import json, sys, time, threading, urllib.request

BASE = "http://10.100.10.2:8000/v1"
MODEL = "nemotron-omni-a4q"

def stream_completion(messages, max_tokens, temperature=0.0):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
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
                c = delta.get("content") or delta.get("reasoning_content")
                if c:
                    if t_first is None:
                        t_first = time.time()
                    text.append(c)
    t_end = time.time()
    comp = usage["completion_tokens"] if usage else None
    ttft = (t_first - t0) if t_first else None
    decode_s = (t_end - t_first) if t_first else None
    decode_tps = comp / decode_s if (comp and decode_s and decode_s > 0) else None
    return {"text": "".join(text), "usage": usage, "ttft": ttft,
            "decode_s": decode_s, "decode_tps": decode_tps,
            "total_s": t_end - t0}

def bench_concurrent(messages, max_tokens, n, temperature=0.0):
    results = [None] * n
    def worker(i):
        results[i] = stream_completion(messages, max_tokens, temperature)
    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    total_comp = sum(r["usage"]["completion_tokens"] for r in results)
    # aggregate decode tok/s: total completion tokens / (wall - mean ttft)
    ttfts = [r["ttft"] for r in results]
    mean_ttft = sum(ttfts) / len(ttfts)
    agg = total_comp / (wall - min(ttfts))
    return {"wall": wall, "total_comp": total_comp, "mean_ttft": mean_ttft,
            "agg_decode_tps": agg,
            "mean_req_tps": sum(r["decode_tps"] for r in results) / n,
            "per_req_tps": [round(r["decode_tps"], 2) for r in results]}

SHORT_PROMPT = ("You are asked to write a detailed technical explanation. "
    "Explain how a modern operating system scheduler decides which process to run next. "
    "Cover the following aspects in order: the distinction between CPU-bound and I/O-bound "
    "processes, the role of time slices and preemption, how priority levels and nice values "
    "influence selection, the idea behind multi-level feedback queues, how completely fair "
    "schedulers use virtual runtime and red-black trees, what happens on a context switch "
    "including register state and TLB effects, how multicore systems handle load balancing "
    "and processor affinity, the interaction between the scheduler and power management "
    "such as frequency scaling, real-time scheduling classes like FIFO and round robin, "
    "and finally how containers and cgroups constrain scheduling decisions. Be thorough, "
    "use concrete examples with plausible numbers, and keep a neutral technical tone "
    "throughout the entire answer without bullet lists, writing in flowing paragraphs. " * 2)

FILLER_SENT = ("The quarterly logistics review noted that warehouse throughput in the "
    "northern distribution corridor improved after the conveyor retrofit, while the "
    "southern hub continued to report intermittent scanner faults during peak loading "
    "windows, prompting a follow-up audit of the maintenance schedule and staffing plan. ")

def build_passkey_prompt(target_tokens_words, passkeys):
    # passkeys: list of (fraction, text)
    n_sent = target_tokens_words // len(FILLER_SENT.split())
    sents = [FILLER_SENT] * n_sent
    for frac, pk in sorted(passkeys, reverse=True):
        idx = int(len(sents) * frac)
        sents.insert(idx, pk + " ")
    return ("Below is a long operations report. Read it carefully; hidden inside are "
            "one or more passkeys.\n\n" + "".join(sents) +
            "\n\nQuestion: First list every passkey mentioned in the report above, exactly as "
            "stated. Then write a summary of the report in about 150 words.")

MATH = [
    ("A farmer has 17 sheep. He buys 3 dozen more, then sells 15. How many sheep does he have now?", "38"),
    ("Tickets cost $12 for adults and $7 for children. A family buys 2 adult and 3 child tickets. What is the total cost in dollars?", "45"),
    ("A train travels 240 km in 3 hours. At the same speed, how many km does it travel in 5 hours?", "400"),
    ("Sara had 45 stickers. She gave 1/5 of them to her brother and then bought 8 more. How many stickers does she have?", "44"),
    ("A rectangle is 9 cm long and 4 cm wide. What is its perimeter in cm?", "26"),
    ("Tom reads 24 pages per day. How many pages does he read in 2 weeks?", "336"),
    ("A shop sells pencils in packs of 6. If a school needs 150 pencils, how many packs must it buy?", "25"),
    ("Lena earns $18 per hour and works 7.5 hours. How much does she earn in dollars?", "135"),
    ("A tank holds 500 liters. It is 3/4 full, then 95 liters are drained. How many liters remain?", "280"),
    ("Three friends split a $96 bill equally and each adds a $4 tip. How much does each person pay in dollars?", "36"),
]

def run_math():
    score = 0
    answers = []
    for q, a in MATH:
        r = stream_completion([
            {"role": "system", "content": "Answer the math problem. End your reply with 'ANSWER: <number>' on its own line."},
            {"role": "user", "content": q}], 512)
        txt = r["text"]
        got = None
        for line in txt.strip().splitlines()[::-1]:
            if "ANSWER" in line.upper():
                got = "".join(c for c in line.split(":")[-1] if c.isdigit() or c == ".").rstrip(".")
                break
        ok = (got is not None and float(got or "nan") == float(a))
        score += ok
        answers.append({"q": q[:40], "expect": a, "got": got, "ok": ok})
    return score, answers

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "short":
        msgs = [{"role": "user", "content": SHORT_PROMPT}]
        print("C1 warm-up run (JIT trigger)...")
        w = stream_completion(msgs, 64)
        print("warmup ok, ttft", round(w["ttft"], 2))
        r = stream_completion(msgs, 512)
        print("SHORT C1:", json.dumps({k: round(v, 3) if isinstance(v, float) else v
              for k, v in r.items() if k != "text"}))
        print("sample:", r["text"][:200].replace("\n", " "))
        c8 = bench_concurrent(msgs, 512, 8)
        print("SHORT C8:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c8.items()}))
    elif mode == "long":
        prompt = build_passkey_prompt(52600,
            [(0.25, "The passkey is 3187."), (0.75, "The second passkey is 9254.")])
        print("prompt words:", len(prompt.split()))
        r = stream_completion([{"role": "user", "content": prompt}], 256)
        u = r["usage"]
        print("LONG C1:", json.dumps({"prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "ttft": round(r["ttft"], 2), "decode_tps": round(r["decode_tps"], 2)}))
        print("passkey 3187 found:", "3187" in r["text"])
        print("passkey 9254 found:", "9254" in r["text"])
        print("reply:", r["text"][:400].replace("\n", " "))
    elif mode == "math":
        s, ans = run_math()
        print(f"MATH {s}/10")
        for a in ans:
            print(" ", a)
    elif mode == "conc":
        # concurrency sweep: C in {1,2,4,8,12,16,20}, ~200-tok prompt, max_tokens 384, temp 0.7
        out_path = sys.argv[2]
        msgs = [{"role": "user", "content": SHORT_PROMPT}]
        stream_completion(msgs, 32)  # warm
        sweep = []
        for c in (1, 2, 4, 8, 12, 16, 20):
            r = bench_concurrent(msgs, 384, c, temperature=0.7)
            row = {"C": c, "agg_decode_tps": round(r["agg_decode_tps"], 2),
                   "mean_req_tps": round(r["mean_req_tps"], 2),
                   "mean_ttft": round(r["mean_ttft"], 3),
                   "wall": round(r["wall"], 2), "total_comp": r["total_comp"]}
            sweep.append(row)
            print("CONC:", json.dumps(row), flush=True)
        json.dump(sweep, open(out_path, "w"), indent=1)
    elif mode == "ctx":
        # context sweep: args = out.json  token_targets...
        out_path = sys.argv[2]
        targets = [int(x) for x in sys.argv[3:]]
        WORDS_PER_TOK = 52638 / 60018  # measured on this corpus/tokenizer
        sweep = []
        for tgt in targets:
            words = int((tgt - 120) * WORDS_PER_TOK)  # 120-tok headroom for template/question
            prompt = build_passkey_prompt(words, [(0.75, "The passkey is 3187.")])
            r = stream_completion([{"role": "user", "content": prompt}], 192)
            u = r["usage"]
            row = {"target": tgt, "prompt_tokens": u["prompt_tokens"],
                   "completion_tokens": u["completion_tokens"],
                   "ttft_s": round(r["ttft"], 2),
                   "decode_tps": round(r["decode_tps"], 2) if r["decode_tps"] else None,
                   "passkey_ok": "3187" in r["text"],
                   "reply_head": r["text"][:120].replace("\n", " ")}
            sweep.append(row)
            print("CTX:", json.dumps(row), flush=True)
        json.dump(sweep, open(out_path, "w"), indent=1)
