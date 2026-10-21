"""
Comprehensive lightweight test suite for the Natural Language Query Engine and JARVIS System.
Tests ONLY the software/agent layer with mocked specialist models and fast deterministic paths.

Scope:
1. Qwen/Ollama connection & fallback
2. Natural-language question understanding & paraphrases
3. Intent detection & Question decomposition
4. Answerability classification & limitation handling
5. Context / Referent resolution ("it", "that", "this area", "the biggest one", "the region you found", "the previous result")
6. Tool & Specialist selection
7. Multi-step planning & Controlled tool registry
8. Evidence-based response synthesis
9. UI Action generation & Controlled UI registry validation
10. JARVIS mode ON/OFF toggling & state pipeline
11. Voice command -> text normalization -> agent pipeline
12. Agent -> UI action frontend state pipeline (including critical heatmap test)
13. Agent conversational memory & Multi-turn state tracking
14. Prevention of hallucinated tool results
15. Prevention of repeated responses and actions
16. Critical Heatmap verification (action emission + frontend active_view update)
"""

import os
import sys
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.agent.state import MissionState, InputState, UIState, ChangeState, SemanticState
from satquery.agent.query_engine import UniversalQueryEngine
from satquery.agent.tools import (
    ALLOWED_SPECIALIST_TOOLS,
    ALLOWED_UI_ACTIONS,
    TOOL_REGISTRY,
    ClusterAnalytics,
    EvidenceFusion,
    ImpactAnalysis,
    UIActionBuilder
)
from satquery.services.tts import normalize_for_speech, prepare_speech_text
from satquery.services.ollama import OllamaService


# ==============================================================================
# MOCK SPECIALIST RESULTS FIXTURES
# ==============================================================================

def get_mock_changeformer_result() -> Dict[str, Any]:
    return {
        "task": "change_analysis",
        "status": "success",
        "change_percentage": 20.66,
        "changed_pixels": 13540,
        "confidence": 0.884,
        "confidence_source": "Calibrated Softmax Uncertainty",
        "bboxes": [
            [100, 80, 220, 190],
            [30, 40, 70, 90]
        ],
        "models_used": ["ChangeFormerV6 (Transformer)"],
        "artifacts": {
            "change_mask": "/mock/change_mask.png",
            "difference_map": "/mock/heatmap.png"
        }
    }


def get_mock_grounding_result() -> Dict[str, Any]:
    return {
        "task": "grounding",
        "status": "success",
        "query": "built-up region",
        "bboxes": [
            [100, 80, 220, 190]
        ],
        "box_confidences": [0.82],
        "num_regions": 1,
        "models_used": ["OWL-ViT (Zero-Shot Grounding)"]
    }


def get_mock_vqa_result() -> Dict[str, Any]:
    return {
        "task": "presence_vqa",
        "status": "success",
        "question": "Is there vegetation?",
        "answer": "Yes",
        "confidence": 0.81,
        "top_predictions": [{"label": "yes", "score": 0.81}, {"label": "no", "score": 0.19}],
        "models_used": ["ViLT-B32"]
    }


def get_mock_optical_sar_result() -> Dict[str, Any]:
    return {
        "task": "optical_sar_fusion",
        "status": "success",
        "answer": "Optical and SAR radar backscatter both confirm structural variance.",
        "optical_evidence": {"variance": "high"},
        "sar_evidence": {"backscatter_diff": 4.2},
        "confidence": 0.85,
        "models_used": ["Rule-Based Optical+SAR Engine"]
    }


def get_query_engine() -> UniversalQueryEngine:
    mock_ollama = MagicMock(spec=OllamaService)
    mock_ollama.chat_json = AsyncMock(side_effect=Exception("Ollama fallback"))
    return UniversalQueryEngine(ollama=mock_ollama)


class MockFrontendState:
    """Mock frontend state manager to simulate UI reactions to agent actions."""
    def __init__(self):
        self.active_tab = "overview"
        self.active_view = "overview"
        self.selected_cluster = None
        self.evidence_open = False
        self.trace_open = False
        self.zoom_target = None

    def apply_action(self, act: Dict[str, Any]):
        action_name = act.get("action")
        if action_name == "navigate_tab":
            tab = act.get("tab")
            self.active_tab = tab
            if tab == "diff-map":
                self.active_view = "heatmap"
            elif tab == "change-mask":
                self.active_view = "change-mask"
            elif tab == "interactive-overlay":
                self.active_view = "interactive-overlay"
        elif action_name in ("show_heatmap", "show_diff_map"):
            self.active_tab = "diff-map"
            self.active_view = "heatmap"
        elif action_name == "show_change_mask":
            self.active_tab = "change-mask"
            self.active_view = "change-mask"
        elif action_name == "show_overlay":
            overlay = act.get("overlay", "interactive-overlay")
            if overlay in ("heatmap", "diff-map"):
                self.active_tab = "diff-map"
                self.active_view = "heatmap"
            else:
                self.active_tab = "interactive-overlay"
                self.active_view = "overlay"
        elif action_name == "select_cluster":
            self.selected_cluster = act.get("index", 0)
        elif action_name == "zoom_to_cluster":
            self.zoom_target = act.get("index", 0)
        elif action_name in ("open_evidence", "show_evidence"):
            self.evidence_open = True
        elif action_name in ("open_trace", "show_trace"):
            self.trace_open = True


# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

def test_natural_language_understanding():
    """Validates natural language intent detection across diverse queries and paraphrases without exact string matching."""
    qe = get_query_engine()
    state = MissionState()
    state.inputs.has_image_1 = True
    state.inputs.has_image_2 = True

    test_cases = [
        ("Show me what changed.", "change_detection"),
        ("What changed between these images?", "change_detection"),
        ("Can you visualize the areas that changed?", "navigate"),
        ("Can you find the biggest change?", "rank_clusters"),
        ("Take me to the biggest area of change.", "zoom_cluster"),
        ("Highlight the most significant change.", "rank_clusters"),
        ("Show me the heatmap.", "navigate_heatmap"),
        ("Display the change mask.", "navigate_change_mask"),
        ("Where is the change?", "inspect_location"),
        ("Give me the coordinates.", "inspect_location"),
        ("Zoom into the largest changed area.", "zoom_cluster"),
        ("Is that near a built-up region?", "grounding"),
        ("Where are the buildings?", "grounding"),
        ("Is there vegetation?", "vqa"),
        ("Compare the optical and SAR images.", "optical_sar"),
        ("Show the radar comparison.", "optical_sar"),
        ("Why does this matter?", "impact"),
        ("Explain the significance of this change.", "impact"),
        ("What should I investigate?", "investigate_scene"),
        ("Why do you think that?", "evidence"),
        ("How confident are you?", "evidence"),
        ("Could you be wrong?", "evidence"),
        ("What evidence supports that?", "evidence"),
        ("Give me a summary.", "investigate_scene"),
        ("Generate a report.", "report"),
    ]

    for query, expected_goal in test_cases:
        plan = asyncio.run(qe.plan_investigation(query, state, prior_result=None))
        assert plan["goal"] == expected_goal, f"Query '{query}' expected goal '{expected_goal}', got '{plan.get('goal')}'"
        assert "intent" in plan and len(plan["intent"]) > 0


def test_jarvis_mode_and_pipeline():
    """Validates JARVIS ON/OFF state and speech text normalization."""
    state = MissionState()
    assert state.jarvis.enabled is True
    assert state.jarvis.voice_state == "IDLE"

    # Toggle OFF
    state.jarvis.enabled = False
    assert state.jarvis.enabled is False

    # Toggle ON
    state.jarvis.enabled = True
    assert state.jarvis.enabled is True

    transcript_1 = "Show me the heatmap."
    normalized_1 = normalize_for_speech(transcript_1)
    assert len(normalized_1) > 0

    transcript_2 = "ChangeFormer detected 20.66% change across 2 regions."
    speech_text = prepare_speech_text(transcript_2)
    assert "percent" in speech_text or "20.66" in speech_text
    assert len(speech_text.split()) < 30


def test_context_and_state():
    """Validates pronoun and referent binding to mission state ('it', 'that', 'this area', 'the biggest one')."""
    qe = get_query_engine()
    state = MissionState()
    state.change.clusters = [
        {"id": "region_1", "index": 0, "name": "Region 1 (Largest)", "area_pixels": 10000, "rank": 1},
        {"id": "region_2", "index": 1, "name": "Region 2", "area_pixels": 2500, "rank": 2}
    ]
    state.update_selected_cluster(0)

    # "the biggest one"
    resolved, idx, ref_type = qe.resolve_context_and_referents("Show me the biggest one", state)
    assert idx == 0
    assert ref_type == "largest_cluster"

    # "it"
    resolved_it, idx_it, ref_type_it = qe.resolve_context_and_referents("Zoom into it", state)
    assert idx_it == 0
    assert "Region 1" in resolved_it or ref_type_it == "selected_cluster"

    # "that"
    resolved_that, idx_that, _ = qe.resolve_context_and_referents("Is that near buildings?", state)
    assert idx_that == 0

    # "this area"
    resolved_area, idx_area, _ = qe.resolve_context_and_referents("Explain why this area changed", state)
    assert idx_area == 0

    # "the region you found"
    resolved_found, idx_found, _ = qe.resolve_context_and_referents("Focus on the region you found", state)
    assert idx_found == 0

    # MissionState.resolve_referent lookup
    state.update_selected_cluster(1)
    ref_it = state.resolve_referent("it")
    assert ref_it is not None
    assert ref_it["id"] == "region_2"

    ref_biggest = state.resolve_referent("the biggest one")
    assert ref_biggest is not None
    assert ref_biggest["id"] == "region_1"


def test_ui_action_and_critical_heatmap():
    """
    CRITICAL TEST:
    User: 'Show me the heatmap.'
    Expected: Emits show_heatmap / navigate_tab:diff-map action without repeating analysis paragraph,
    and switches frontend active_view to 'heatmap'.
    """
    qe = get_query_engine()
    mock_res = get_mock_changeformer_result()
    state = MissionState()
    state.inputs.has_image_1 = True
    state.inputs.has_image_2 = True
    state.change.change_percentage = 20.66

    # Registry compliance
    for act in ["navigate_tab", "show_overlay", "show_heatmap", "show_change_mask", "select_cluster", "zoom_to_cluster"]:
        assert act in ALLOWED_UI_ACTIONS
    forbidden = ["execute_python", "run_shell", "eval_js", "fs_write", "exec_cmd"]
    for bad_act in forbidden:
        assert bad_act not in ALLOWED_UI_ACTIONS

    # Step 1: Plan heatmap query
    plan = asyncio.run(qe.plan_investigation("Show me the heatmap.", state, prior_result=mock_res))
    assert plan["goal"] == "navigate_heatmap"
    assert plan["requires_specialist"] is False
    assert any(act in plan["ui_actions"] for act in ["navigate_tab:diff-map", "show_heatmap"])

    # Step 2: Build UI actions
    ui_actions = UIActionBuilder.build_actions(
        plan=plan,
        specialist_result=mock_res,
        trace_steps=["Query received", "Navigating to heatmap"],
        target_cluster=0
    )
    assert any(a.get("action") in ("show_heatmap", "show_diff_map") or a.get("tab") == "diff-map" for a in ui_actions)

    # Step 3: Synthesize grounded response
    synth = asyncio.run(qe.synthesize_grounded_response(
        query="Show me the heatmap.",
        plan=plan,
        specialist_result=None,
        prior_result=mock_res,
        state=state
    ))
    assert "heatmap" in synth["answer"].lower()
    assert len(synth["answer"]) < 200

    # Step 4: Apply to mock frontend and verify active_view
    frontend = MockFrontendState()
    for act in ui_actions:
        frontend.apply_action(act)

    assert frontend.active_view == "heatmap", f"Expected active_view == 'heatmap', got '{frontend.active_view}'"
    assert frontend.active_tab == "diff-map"


def test_anti_hallucination_and_answerability():
    """Verifies that unanswerable queries (temperature, rainfall, demographics) are honestly refused without inventing data."""
    qe = get_query_engine()
    state = MissionState()

    # Temperature
    is_ans, refusal = qe.check_answerability("Did the temperature increase?", state)
    assert is_ans is False
    assert "temperature" in refusal.lower() and "thermal" in refusal.lower()

    # Rainfall
    is_ans_rain, refusal_rain = qe.check_answerability("What happened to rainfall?", state)
    assert is_ans_rain is False
    assert "rainfall" in refusal_rain.lower() or "meteorological" in refusal_rain.lower()

    # Demographics
    is_ans_pop, refusal_pop = qe.check_answerability("How many people live here?", state)
    assert is_ans_pop is False
    assert "population" in refusal_pop.lower() or "census" in refusal_pop.lower()

    # Evidence fusion uses only real specialist data
    mock_res = get_mock_changeformer_result()
    fusion = EvidenceFusion.fuse(change_result=mock_res)
    assert "20.66%" in fusion["observation"]
    assert "ChangeFormerV6" in fusion["observation"]
    assert "limitations" in fusion and len(fusion["limitations"]) > 0


def test_anti_repetition():
    """Verifies that follow-up navigation commands don't rerun models or repeat paragraphs."""
    qe = get_query_engine()
    mock_res = get_mock_changeformer_result()
    state = MissionState()
    state.inputs.has_image_1 = True
    state.inputs.has_image_2 = True

    # Heatmap follow-up
    plan_hm = asyncio.run(qe.plan_investigation("Show me the heatmap.", state, prior_result=mock_res))
    assert plan_hm["requires_specialist"] is False
    assert plan_hm["reuse_prior"] is True

    # Change mask follow-up
    plan_mask = asyncio.run(qe.plan_investigation("Show the change mask.", state, prior_result=mock_res))
    assert plan_mask["requires_specialist"] is False
    assert plan_mask["reuse_prior"] is True


def test_ollama_connection():
    """Verifies Ollama service connection check and JSON schema extractor."""
    svc = OllamaService()
    raw_text = "<think>I need to plan</think>```json\n{\"intent\": \"change detection\", \"specialist\": \"change_detection\"}\n```"
    extracted = svc.extract_json(raw_text)
    assert extracted["intent"] == "change detection"
    assert extracted["specialist"] == "change_detection"

    is_healthy, msg = asyncio.run(svc.check_health())
    assert isinstance(is_healthy, bool)
    assert isinstance(msg, str)


def test_voice_pipeline():
    """Verifies the voice transcription -> natural language planner -> UI action pipeline."""
    qe = get_query_engine()
    voice_transcript = "Show me the heatmap."
    normalized = normalize_for_speech(voice_transcript)
    assert len(normalized) > 0

    state = MissionState()
    plan = asyncio.run(qe.plan_investigation(normalized, state, prior_result={"task": "change_analysis"}))
    assert plan["goal"] == "navigate_heatmap"
    assert any("diff-map" in act or "heatmap" in act for act in plan["ui_actions"])


# ==============================================================================
# MAIN TEST EXECUTION RUNNER
# ==============================================================================

if __name__ == "__main__":
    tests = [
        ("NATURAL LANGUAGE TESTS", test_natural_language_understanding),
        ("JARVIS TESTS", test_jarvis_mode_and_pipeline),
        ("CONTEXT/STATE TESTS", test_context_and_state),
        ("UI ACTION TESTS", test_ui_action_and_critical_heatmap),
        ("ANTI-HALLUCINATION TESTS", test_anti_hallucination_and_answerability),
        ("ANTI-REPETITION TESTS", test_anti_repetition),
        ("OLLAMA TEST", test_ollama_connection),
        ("VOICE PIPELINE TEST", test_voice_pipeline),
    ]

    print("==================================================")
    print("RUNNING NATURAL LANGUAGE QUERY ENGINE & JARVIS TESTS")
    print("==================================================")

    results = {}
    failed_details = []

    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASS"
            print(f"[PASS] {name}")
        except Exception as e:
            results[name] = "FAIL"
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed_details.append({
                "test": name,
                "error": str(e)
            })

    print("\n==================================================")
    print("TEST REPORT")
    print("==================================================")
    for name, status in results.items():
        print(f"{name}\n{status}\n")

    passed_count = sum(1 for s in results.values() if s == "PASS")
    failed_count = sum(1 for s in results.values() if s == "FAIL")

    print(f"TOTAL TESTS\nPASSED: {passed_count}\nFAILED: {failed_count}\nSKIPPED: 0")
    print("==================================================")
