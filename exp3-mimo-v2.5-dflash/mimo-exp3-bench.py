#!/usr/bin/env python3
"""Exp3 bench: C1 + C8 streaming decode tok/s against localhost:8000, plus 10-problem eval."""
import json, sys, time, threading, urllib.request

BASE = "http://127.0.0.1:8000/v1"
MODEL = "mimo-v25"

def stream_completion(prompt, max_tokens=512, temperature=0.0, chat=False):
    if chat:
        url = f"{BASE}/chat/completions"
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": temperature,
                "stream": True, "stream_options": {"include_usage": True}}
    else:
        url = f"{BASE}/completions"
        body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); tfirst = None; tlast = None
    text = []; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = (ch.get("delta") or {}).get("content") if chat else ch.get("text")
                if delta:
                    if tfirst is None:
                        tfirst = time.time()
                    tlast = time.time()
                    text.append(delta)
    return {"text": "".join(text), "usage": usage, "t0": t0,
            "tfirst": tfirst, "tlast": tlast}

def bench(conc, prompt, max_tokens=512):
    results = [None] * conc
    def worker(i):
        p = prompt if conc == 1 else f"{prompt} (variant {i}: use a different example)"
        results[i] = stream_completion(p, max_tokens=max_tokens)
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    total_out = sum(r["usage"]["completion_tokens"] for r in results if r and r["usage"])
    # per-request decode rate: completion_tokens / (tlast - tfirst)
    rates = []
    for r in results:
        if r and r["usage"] and r["tfirst"] and r["tlast"] and r["tlast"] > r["tfirst"]:
            rates.append((r["usage"]["completion_tokens"] - 1) / (r["tlast"] - r["tfirst"]))
    agg = total_out / wall
    return {"conc": conc, "wall_s": round(wall, 2), "total_completion_tokens": total_out,
            "agg_tok_s": round(agg, 2),
            "mean_per_req_decode_tok_s": round(sum(rates) / len(rates), 2) if rates else None,
            "sample": results[0]["text"][:400] if results[0] else None}

EVAL = [
    ("A farmer has 17 sheep. All but 9 run away. How many sheep are left? Answer with just the number.", "9"),
    ("What is 37 * 24? Answer with just the number.", "888"),
    ("Tom has 3 boxes with 12 apples each. He gives away 7 apples. How many apples does he have left? Answer with just the number.", "29"),
    ("A train travels 60 km in 45 minutes. What is its speed in km/h? Answer with just the number.", "80"),
    ("What is the sum of the first 10 positive integers? Answer with just the number.", "55"),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines take to make 100 widgets? Answer with just the number.", "5"),
    ("Sarah is twice as old as her brother. In 6 years the sum of their ages will be 42. How old is Sarah now? Answer with just the number.", "20"),
    ("What is 15% of 240? Answer with just the number.", "36"),
    ("A rectangle has perimeter 36 and length 10. What is its area? Answer with just the number.", "80"),
    ("What is 2^10? Answer with just the number.", "1024"),
]

def run_eval():
    out = []
    for q, gold in EVAL:
        r = stream_completion(q, max_tokens=2048, temperature=0.0, chat=True)
        ans = r["text"]
        # take the last number-ish token in the reply
        import re
        nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
        got = nums[-1] if nums else ""
        ok = False
        try:
            ok = abs(float(got) - float(gold)) < 1e-6
        except ValueError:
            pass
        out.append({"q": q[:50], "gold": gold, "got": got, "ok": ok, "full": ans[-2000:]})
    score = sum(1 for o in out if o["ok"])
    return {"score": f"{score}/10", "details": out}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if mode == "bench":
        prompt = "Explain in detail how a refrigerator works, covering the vapor-compression cycle."
        r1 = bench(1, prompt)
        print("C1:", json.dumps(r1, indent=1))
        r8 = bench(8, prompt)
        print("C8:", json.dumps(r8, indent=1))
    elif mode == "eval":
        r = run_eval()
        print(json.dumps(r, indent=1))
    elif mode == "coherence":
        r = stream_completion(
            "Write a coherent 150-word paragraph about the history of the Silk Road.",
            max_tokens=400, temperature=0.0, chat=True)
        print(r["text"])
        print("\nUSAGE:", r["usage"])

def sweep_conc():
    prompt = "Explain in detail how a refrigerator works, covering the vapor-compression cycle, each component, and the thermodynamics."
    rows = []
    for c in [1, 4, 8, 12, 16, 20]:
        r = bench(c, prompt, max_tokens=256)
        rows.append(r)
        print(f"C{c}: agg={r['agg_tok_s']} per_req={r['mean_per_req_decode_tok_s']} wall={r['wall_s']}")
    json.dump(rows, open("/home/keyspark/mimo-exp3-conc-sweep.json","w"), indent=1)

def sweep_ctx():
    import time
    # build long prompts by repeating a filler paragraph, target token counts
    filler = ("The quick brown fox jumps over the lazy dog near the riverbank while the sun sets slowly behind the distant mountains. " * 20)
    rows = []
    for approx_tok in [6000, 32000, 64000, 128000]:
        # ~0.75 tok/word; filler ~ 260 tokens per copy. estimate copies
        copies = max(1, approx_tok // 260)
        body_prompt = (filler * copies)[: approx_tok * 5]  # char cap safety
        q = body_prompt + "\n\nBased on the text above, reply with exactly one short sentence summarizing it."
        body = {"model": MODEL, "messages": [{"role":"user","content":q}],
                "max_tokens": 64, "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True}}
        req = urllib.request.Request(f"{BASE}/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
        t0=time.time(); tfirst=None; usage=None; txt=[]
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                for line in r:
                    line=line.decode().strip()
                    if not line.startswith("data: "): continue
                    p=line[6:]
                    if p=="[DONE]": break
                    d=json.loads(p)
                    if d.get("usage"): usage=d["usage"]
                    for ch in d.get("choices",[]):
                        dl=(ch.get("delta") or {}).get("content")
                        if dl:
                            if tfirst is None: tfirst=time.time()
                            txt.append(dl)
            ttft = (tfirst-t0) if tfirst else None
            ptok = usage["prompt_tokens"] if usage else None
            ctok = usage["completion_tokens"] if usage else None
            dec_time = (time.time()-tfirst) if tfirst else None
            dec_rate = (ctok-1)/dec_time if dec_time and ctok else None
            row={"target_tok":approx_tok,"prompt_tokens":ptok,"ttft_s":round(ttft,2) if ttft else None,
                 "prefill_tok_s":round(ptok/ttft,1) if ttft and ptok else None,
                 "decode_tok_s":round(dec_rate,2) if dec_rate else None,"coherent_sample":"".join(txt)[:150]}
        except Exception as e:
            row={"target_tok":approx_tok,"error":str(e)[:200]}
        rows.append(row); print(json.dumps(row))
    json.dump(rows, open("/home/keyspark/mimo-exp3-ctx-sweep.json","w"), indent=1)

if __name__ == "__main__" and len(sys.argv)>1 and sys.argv[1]=="conc": sweep_conc()
if __name__ == "__main__" and len(sys.argv)>1 and sys.argv[1]=="ctx": sweep_ctx()
