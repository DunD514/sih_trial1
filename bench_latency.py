"""
SatQuery AI — Latency Benchmark (Post-Optimization)
Tests all major JARVIS query paths and reports response times.
Run with: python bench_latency.py
"""

import asyncio
import time
import httpx
import sys

BASE_URL = "http://127.0.0.1:8080"
IMAGE_1_PATH = "samples_LEVIR/A/test_2_0000_0000.png"
IMAGE_2_PATH = "samples_LEVIR/B/test_2_0000_0000.png"

QUERIES = [
    # (label, query, has_image_2)
    # UI Fast Path
    ("UI: show heatmap",            "show me the heat map",               False),
    ("UI: show change mask",         "show the change mask",               False),
    ("UI: show side by side",        "show side by side",                  False),
    # Generic Scene Description
    ("SCENE cold",                   "What does this image show?",         False),
    ("SCENE warm/cached",            "What does this image show?",         False),
    ("SCENE alt-wording cold",       "Explain the image",                  False),
    ("SCENE alt-wording warm",       "Explain the image",                  False),
    # Change Detection
    ("CHANGE cold",                  "what changed?",                      True),
    ("CHANGE warm/cached",           "what changed?",                      True),
    # VQA
    ("VQA: are there buildings?",    "are there buildings in this image?", False),
    # Unanswerable / Limitations
    ("LIMIT: temperature query",     "what is the temperature?",           False),
]

SEP = "-" * 72


def latency_tag(t: float) -> str:
    if t < 0.5:   return "[FAST]"
    if t < 20.0:  return "[OK]"
    if t < 40.0:  return "[SLOW]"
    return "[VERY SLOW]"


async def check_server() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{BASE_URL}/api/v1/health")
            return r.status_code == 200
    except Exception:
        return False


async def run_query(query: str, has_image_2: bool) -> dict:
    payload = {
        "query": query,
        "has_image_1": True,
        "has_image_2": has_image_2,
        "image_1_path": IMAGE_1_PATH,
        "image_2_path": IMAGE_2_PATH if has_image_2 else None,
        "voice_enabled": False,
    }
    async with httpx.AsyncClient(timeout=180.0) as c:
        t0 = time.perf_counter()
        r = await c.post(f"{BASE_URL}/api/v1/jarvis", json=payload)
        elapsed = time.perf_counter() - t0
        data = r.json()
        return {
            "elapsed": elapsed,
            "intent":  data.get("intent", "?"),
            "answer":  str(data.get("answer", ""))[:80].replace("\n", " "),
            "status_code": r.status_code,
        }


async def main():
    print(f"\nSatQuery AI — Latency Benchmark (Post-Optimization)")
    print(f"Backend : {BASE_URL}")
    print(SEP)

    if not await check_server():
        print(f"ERROR: Backend not reachable at {BASE_URL}")
        sys.exit(1)
    print("Backend: ONLINE\n")

    results = []
    for label, query, has_img2 in QUERIES:
        sys.stdout.write(f"  {label:<42} ... ")
        sys.stdout.flush()
        try:
            r = await run_query(query, has_image_2=has_img2)
            t = r["elapsed"]
            tag = latency_tag(t)
            print(f"{t:6.2f}s  {tag:12s}  [{r['intent']}]  \"{r['answer']}\"")
            results.append((label, t, r["intent"], r["status_code"]))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((label, -1.0, "ERROR", 0))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"{'QUERY':<44} {'TIME':>7}  STATUS")
    print(SEP)
    for label, t, intent, sc in results:
        t_str  = f"{t:6.2f}s" if t >= 0 else "  ERROR"
        status = "OK" if sc == 200 else "FAIL"
        tag    = latency_tag(t) if t >= 0 else "[ERROR]"
        print(f"  {label:<44} {t_str}  {status}  {tag}")

    print(SEP)
    ui_times = [t for l, t, _, sc in results if "UI:" in l and t >= 0]
    ai_times = [t for l, t, _, sc in results if "UI:" not in l and "LIMIT" not in l and t >= 0]
    if ui_times:
        print(f"  UI fast-path avg  : {sum(ui_times)/len(ui_times):.3f}s")
    if ai_times:
        print(f"  AI reasoning avg  : {sum(ai_times)/len(ai_times):.2f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
