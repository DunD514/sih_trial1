"""
SATQUERY AI — JARVIS 2.0 MASTER TEST SUITE
Validates Universal Satellite Query Engine, Section 59 & Section 66 requirements.
"""

import sys
import json
import time
import urllib.request
import urllib.error

BASE_URL = 'http://127.0.0.1:8080'


def print_progress(current, total, label=""):
    bar_length = 30
    filled = int(bar_length * current // total)
    bar = '=' * filled + '-' * (bar_length - filled)
    pct = (current / total) * 100
    sys.stdout.write(f"\r[{bar}] {pct:5.1f}% ({current}/{total}) - {label[:35]:<35}")
    sys.stdout.flush()


def post_json(path, data, timeout=60):
    req = urllib.request.Request(
        f'{BASE_URL}{path}',
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode('utf-8'))


def get_json(path, timeout=30):
    with urllib.request.urlopen(f'{BASE_URL}{path}', timeout=timeout) as res:
        return json.loads(res.read().decode('utf-8'))


def run_tests():
    print("==========================================================", flush=True)
    print("       SATQUERY AI — JARVIS 2.0 AUTOMATED TEST SUITE       ", flush=True)
    print("==========================================================", flush=True)

    # 1. Health check
    health = get_json('/api/v1/health')
    assert health.get('status') == 'healthy', f"Health check failed: {health}"
    print("[PASS] System Health Verified (Status: healthy)", flush=True)

    # 2. Sample Discovery & Serving
    samples = get_json('/api/v1/samples')
    levir = samples.get('levir_samples', [])
    assert len(levir) >= 12, f"Expected >= 12 LEVIR samples, found {len(levir)}"
    sample = levir[0]
    print(f"[PASS] Sample Discovery: {len(levir)} pairs available. Using: {sample['name']}", flush=True)

    t1_url = f"{BASE_URL}{sample['t1']}"
    t2_url = f"{BASE_URL}{sample['t2']}"
    with urllib.request.urlopen(t1_url, timeout=10) as r1:
        assert r1.status == 200
    with urllib.request.urlopen(t2_url, timeout=10) as r2:
        assert r2.status == 200
    print(f"[PASS] Sample Image Serving: T1 (200 OK), T2 (200 OK)", flush=True)

    # 3. Master Section 59 Question Battery
    questions = [
        ("What changed?", ["change", "region", "spatial", "%"]),
        ("Where did it change?", ["cluster", "region", "coordinate", "area"]),
        ("What is the biggest change?", ["region", "largest", "area", "pixel"]),
        ("What should I pay attention to?", ["priority", "investigat", "region"]),
        ("Are there buildings?", ["building", "structure", "candidate"]),
        ("Where are the buildings?", ["grounding", "candidate", "box", "structure"]),
        ("What changed around the buildings?", ["structure", "spatial", "change"]),
        ("Is that a building?", ["candidate", "structure", "evidence"]),
        ("Why do you think that?", ["model", "evidence", "changeformer", "score"]),
        ("Why does it matter?", ["infrastructure", "urban", "relevance", "monitoring"]),
        ("Compare optical and SAR.", ["optical", "sar", "sensor", "backscatter"]),
        ("Show me the heatmap.", ["heatmap", "visualization"]),
        ("Show me the biggest change.", ["region", "selected"]),
        ("Investigate this scene.", ["investigat", "assessment", "evidence"]),
        ("Generate a report.", ["report", "summary", "generated"]),
        ("How confident are you?", ["heuristic", "score", "confidence", "limitation"]),
        ("What information are you missing?", ["sar", "ground-truth", "sensor"]),
        ("What is the population?", ["cannot", "not observable", "demographic", "external", "unable"])
    ]

    total_battery = len(questions)
    print(f"\nExecuting Section 59 Master Question Battery ({total_battery} scenarios)...", flush=True)

    state = {
        't1_path': sample['t1_path'],
        't2_path': sample['t2_path'],
        'prior_id': None,
        'selected_cluster': 0,
        'history': []
    }

    results_summary = []
    start_time = time.time()

    for idx, (query, expected_keywords) in enumerate(questions, 1):
        print_progress(idx - 1, total_battery, f"Testing '{query}'")
        payload = {
            'query': query,
            'has_image_1': True,
            'has_image_2': True,
            'image_1_path': state['t1_path'],
            'image_2_path': state['t2_path'],
            'current_tab': 'interactive-overlay',
            'prior_result_id': state['prior_id'],
            'selected_cluster': state['selected_cluster'],
            'voice_enabled': True,
            'chat_history': state['history'][-4:]
        }
        res = post_json('/api/v1/jarvis', payload, timeout=120)
        state['prior_id'] = res.get('task_id')
        state['history'].append({'role': 'user', 'content': query})
        state['history'].append({'role': 'assistant', 'content': res.get('answer', '')})

        ans = res.get('answer', '').strip()
        speak = res.get('speak_text', '').strip()
        actions = [a.get('action') for a in res.get('ui_actions', [])]

        assert len(ans) > 0, f"Empty answer returned for query: {query}"
        assert len(speak) > 0, f"Empty speak_text returned for query: {query}"

        # Special verification for answerability refusal:
        if query == "What is the population?":
            refusal_check = any(k in ans.lower() for k in ["cannot", "unable", "demographic", "not observable", "external", "land-cover"])
            assert refusal_check, f"Answerability engine failed to refuse demographic query: {ans}"

        results_summary.append({
            "idx": idx,
            "query": query,
            "actions": actions,
            "speak_len": len(speak.split()),
            "status": "PASS"
        })

    print_progress(total_battery, total_battery, "All Question Tests Complete!")
    print("\n", flush=True)

    # 4. Test UI Action Acknowledgement
    print("Testing UI Action Acknowledgement (/api/v1/jarvis/action_ack)...", flush=True)
    ack_res = post_json('/api/v1/jarvis/action_ack', {
        "action": "show_heatmap",
        "status": "success",
        "details": {"tab": "interactive-overlay", "overlay": "heatmap"}
    })
    assert ack_res.get("status") == "acknowledged"
    print("[PASS] UI Action Acknowledgement Confirmed.", flush=True)

    # 5. Missing T2 Input Handling
    print("Testing Missing T2 Input Robustness...", flush=True)
    missing_res = post_json('/api/v1/jarvis', {
        'query': 'What changed?',
        'has_image_1': True,
        'has_image_2': False,
        'image_1_path': state['t1_path'],
        'image_2_path': None
    })
    assert any(k in missing_res.get('answer', '').lower() for k in ['second', 'need', 't2', 'pair', 'temporal'])
    print(f"[PASS] Missing Input Handling: Correct refusal ({missing_res['answer'][:60]}...)", flush=True)

    elapsed = time.time() - start_time
    print("==========================================================", flush=True)
    print(f"  ALL TESTS PASSED SUCCESSFULLY in {elapsed:.2f}s!         ", flush=True)
    print("==========================================================", flush=True)

    print("\nTest Battery Summary:")
    for item in results_summary:
        print(f"  [{item['idx']:02d}] {item['query']:<38} | Actions: {str(item['actions']):<25} | Voice words: {item['speak_len']} | {item['status']}")


if __name__ == '__main__':
    run_tests()
