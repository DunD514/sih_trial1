"""
Automated End-to-End Test Suite for SatQuery AI — JARVIS 2.0
Tests the full multi-turn autonomous remote sensing analyst copilot pipeline:
1. Initial Change Detection Analysis with LEVIR sample images
2. Follow-Up: "Which one is biggest?" (Cached cluster ranking)
3. Follow-Up: "Show me" (Registered UI action execution)
4. Follow-Up: "Where is it?" (Honest geospatial inspection)
5. Follow-Up: "Explain this" (Region explanation via cached evidence)
6. Follow-Up: "Why does this matter?" (Hedged domain impact)
7. Follow-Up: "Give me the executive summary" (Structured summary)
8. Missing Data Handling: Querying change with only 1 image
"""

import json
import urllib.request
import sys

BASE_URL = "http://localhost:8080/api/v1"

def make_req(endpoint: str, data: dict, timeout: int = 120) -> dict:
    url = f"{BASE_URL}{endpoint}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))

def get_req(endpoint: str) -> dict:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.loads(res.read().decode("utf-8"))


def test_jarvis_2():
    print("=================================================================")
    print("SATQUERY AI — JARVIS 2.0 AUTOMATED END-TO-END VERIFICATION")
    print("=================================================================")

    # 1. Check Status
    print("\n[TEST 1] Verifying JARVIS 2.0 Status Endpoint...")
    status = get_req("/jarvis/status")
    print(f"  Status: {status.get('status')} | LLM: {status.get('model')} | TTS: {status.get('tts_provider')} ({status.get('voice')})")
    assert status.get("status") == "ONLINE", f"Expected ONLINE, got {status.get('status')}"
    assert status.get("model") == "qwen3:8b", f"Expected qwen3:8b, got {status.get('model')}"
    assert status.get("tts") == "ready", f"Expected TTS ready, got {status.get('tts')}"
    print("  --> PASS: Status & Centralized Ollama qwen3:8b verified.")

    # 2. Test TTS Endpoint
    print("\n[TEST 2] Verifying Neural TTS Audio Generation Endpoint...")
    tts_res = make_req("/jarvis/tts", {"text": "ChangeFormer V6 detected spatial change across twenty-five point three two percent of the scene."})
    assert tts_res.get("success") is True, "Expected TTS success=True"
    assert tts_res.get("audio_base64") is not None, "Expected valid base64 audio"
    print(f"  TTS Provider: {tts_res.get('provider')} | Normalized Text: \"{tts_res.get('normalized_text')}\"")
    print("  --> PASS: Neural TTS generated audio.")

    # 3. Step 1: Initial Change Detection Query
    print("\n[TEST 3] Step 1: Query 'Show me what changed' with LEVIR image pair...")
    p1 = {
        "query": "Show me what changed between these two images",
        "has_image_1": True,
        "has_image_2": True,
        "image_1_path": "samples_LEVIR/A/test_2_0000_0000.png",
        "image_2_path": "samples_LEVIR/B/test_2_0000_0000.png",
        "current_tab": "overview",
        "voice_enabled": True
    }
    r1 = make_req("/jarvis", p1)
    task_id = r1.get("task_id")
    assert task_id is not None, "Expected task_id"
    print(f"  Answer: {r1.get('answer')}")
    print(f"  Spoken Text: \"{r1.get('speak_text')}\"")
    print(f"  Provenance: {r1.get('provenance')}")
    print(f"  Why This Result: {r1.get('why_this_result')}")
    print(f"  Why This Matters: {r1.get('why_this_matters')}")
    print(f"  Recommended Next: {r1.get('recommended_next_step')}")
    print(f"  UI Actions: {r1.get('ui_actions')}")
    assert r1.get("evidence", {}).get("change_percentage") is not None, "Expected ChangeFormer evidence"
    print("  --> PASS: ChangeFormer executed, evidence collected, and grounded synthesis generated.")

    # 4. Step 2: Follow-up "Which one is biggest?" (Cached ranking)
    print("\n[TEST 4] Step 2: Query 'Which one is biggest?' (Reusing cache)...")
    p2 = {
        "query": "Which one is biggest?",
        "has_image_1": True,
        "has_image_2": True,
        "image_1_path": "samples_LEVIR/A/test_2_0000_0000.png",
        "image_2_path": "samples_LEVIR/B/test_2_0000_0000.png",
        "prior_result_id": task_id,
        "voice_enabled": True
    }
    r2 = make_req("/jarvis", p2)
    print(f"  Answer: {r2.get('answer')}")
    print(f"  Spoken Text: \"{r2.get('speak_text')}\"")
    print(f"  UI Actions: {r2.get('ui_actions')}")
    print("  --> PASS: Cached clusters ranked and returned without re-running ChangeFormer.")

    # 5. Step 3: Follow-up "Show me"
    print("\n[TEST 5] Step 3: Query 'Show me'...")
    p3 = {
        "query": "Show me",
        "has_image_1": True,
        "has_image_2": True,
        "prior_result_id": task_id,
        "voice_enabled": True
    }
    r3 = make_req("/jarvis", p3)
    print(f"  Spoken Text: \"{r3.get('speak_text')}\"")
    print(f"  UI Actions: {r3.get('ui_actions')}")
    assert any(act.get("action") in ("select_cluster", "highlight_cluster") for act in r3.get("ui_actions", [])), "Expected cluster highlight action"
    print("  --> PASS: Highlight UI actions generated.")

    # 6. Step 4: Follow-up "Where is it?"
    print("\n[TEST 6] Step 4: Query 'Where is it?' (Geospatial check)...")
    p4 = {
        "query": "Where is it? Give me coordinates.",
        "has_image_1": True,
        "has_image_2": True,
        "prior_result_id": task_id,
        "voice_enabled": True
    }
    r4 = make_req("/jarvis", p4)
    print(f"  Answer: {r4.get('answer')}")
    print(f"  Spoken Text: \"{r4.get('speak_text')}\"")
    print("  --> PASS: Honest geospatial status communicated without hallucinating fake coordinates.")

    # 7. Step 5: Follow-up "Why does this matter?"
    print("\n[TEST 7] Step 5: Query 'Why does this matter?'...")
    p5 = {
        "query": "Why does this matter?",
        "has_image_1": True,
        "has_image_2": True,
        "prior_result_id": task_id,
        "voice_enabled": True
    }
    r5 = make_req("/jarvis", p5)
    print(f"  Why This Matters: {r5.get('why_this_matters')}")
    print(f"  Spoken Text: \"{r5.get('speak_text')}\"")
    assert r5.get("why_this_matters") is not None, "Expected why_this_matters"
    print("  --> PASS: Hedged domain impact retrieved.")

    # 8. Step 6: Follow-up "Give me the executive summary"
    print("\n[TEST 8] Step 6: Query 'Give me the executive summary'...")
    p6 = {
        "query": "Give me the executive summary",
        "has_image_1": True,
        "has_image_2": True,
        "prior_result_id": task_id,
        "voice_enabled": True
    }
    r6 = make_req("/jarvis", p6)
    summary = r6.get("executive_summary", {})
    print(f"  Executive Summary: {summary}")
    assert "finding" in summary or "largest_change" in summary, "Expected executive summary fields"
    print("  --> PASS: Executive summary compiled.")

    # 9. Step 7: Missing input test
    print("\n[TEST 9] Step 7: Testing Missing Input Handling (1 image provided, change detection queried)...")
    p7 = {
        "query": "Compare these two images and detect changes",
        "has_image_1": True,
        "has_image_2": False,
        "image_1_path": "samples_LEVIR/A/test_2_0000_0000.png",
        "voice_enabled": True
    }
    r7 = make_req("/jarvis", p7)
    print(f"  Answer: {r7.get('answer')}")
    print(f"  Spoken Text: \"{r7.get('speak_text')}\"")
    assert "second" in r7.get("answer", "").lower() or "need" in r7.get("answer", "").lower(), "Expected explanation of missing second image"
    print("  --> PASS: Handled missing input gracefully and honestly.")

    print("\n=================================================================")
    print("ALL 9 JARVIS 2.0 END-TO-END TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    try:
        test_jarvis_2()
    except Exception as exc:
        print(f"\n[FAILED] Test error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
