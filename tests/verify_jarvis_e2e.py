import requests
import time
import json

base_url = "http://127.0.0.1:8080"

print("--- 1. Checking Health ---")
r = requests.get(f"{base_url}/api/v1/health")
print("Health:", r.status_code, r.json().get("status"))

print("\n--- 2. Fetching Benchmark Samples ---")
samples = requests.get(f"{base_url}/api/v1/samples").json()
levir = samples["levir_samples"][0]
t1_path = levir["t1_path"]
t2_path = levir["t2_path"]
print(f"Loaded sample pair: {t1_path} & {t2_path}")

print("\n--- 3. Testing 'show me the heat map' (UI Action) ---")
# First request (cold)
t0 = time.time()
r_hm = requests.post(f"{base_url}/api/v1/jarvis", json={
    "query": "show me the heat map",
    "has_image_1": True,
    "has_image_2": True,
    "image_1_path": t1_path,
    "image_2_path": t2_path
}).json()
d_hm_cold = time.time() - t0
print(f"HEATMAP TEST (cold): duration={d_hm_cold:.3f}s")

# Second request (warm)
t0 = time.time()
r_hm2 = requests.post(f"{base_url}/api/v1/jarvis", json={
    "query": "show me the heat map",
    "has_image_1": True,
    "has_image_2": True,
    "image_1_path": t1_path,
    "image_2_path": t2_path
}).json()
d_hm = time.time() - t0
print(f"HEATMAP TEST (warm): duration={d_hm:.3f}s")
print(f"Intent: {r_hm.get('intent')}")
print(f"UI Action: {r_hm.get('ui_action')}")
print(f"Response: {r_hm.get('response')}")
assert r_hm.get("intent") == "UI_SHOW_HEATMAP"
assert r_hm.get("ui_action") == {"type": "SELECT_VIEW", "view": "difference-heatmap"}
assert d_hm < 1.0, f"Heatmap latency exceeded 1.0s: {d_hm:.3f}s"

print("\n--- 4. Testing 'show the change mask' (UI Action) ---")
t0 = time.time()
r_cm = requests.post(f"{base_url}/api/v1/jarvis", json={
    "query": "show the change mask",
    "has_image_1": True,
    "has_image_2": True,
    "image_1_path": t1_path,
    "image_2_path": t2_path
}).json()
d_cm = time.time() - t0
print(f"CHANGE MASK TEST: duration={d_cm:.3f}s")
print(f"Intent: {r_cm.get('intent')}")
print(f"UI Action: {r_cm.get('ui_action')}")
assert r_cm.get("intent") == "UI_SHOW_CHANGE_MASK"
assert r_cm.get("ui_action") == {"type": "SELECT_VIEW", "view": "change-mask"}
assert d_cm < 1.0

print("\n--- 5. Testing 'locate structures' (OWL-ViT Grounding) ---")
t0 = time.time()
r_loc = requests.post(f"{base_url}/api/v1/jarvis", json={
    "query": "locate structures",
    "has_image_1": True,
    "has_image_2": False,
    "image_1_path": t1_path
}).json()
d_loc = time.time() - t0
print(f"LOCATE STRUCTURES TEST: duration={d_loc:.3f}s")
print(f"Intent: {r_loc.get('intent')}")
print(f"Answer: {r_loc.get('answer')}")
print(f"Num regions: {r_loc.get('evidence', {}).get('num_regions')}")
assert r_loc.get("intent") in ["OBJECT_LOCATION", "OBJECT_DETECTION"]
assert "Error" not in r_loc.get("answer", "")
assert "logger" not in r_loc.get("answer", "")

print("\n--- ALL E2E ENDPOINT VERIFICATIONS PASSED ---")
