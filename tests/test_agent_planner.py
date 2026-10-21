"""
Unit tests for NaturalLanguagePlanner, Intent Taxonomy, and Query Routing.
Verifies all 11 required query routing cases from the architectural specification.
"""

import sys
import os

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.agent.context import SceneContext
from satquery.agent.planner import NaturalLanguagePlanner, INTENT_TAXONOMY


def test_11_required_routing_scenarios():
    ctx_dual = SceneContext()
    ctx_dual.has_image_1 = True
    ctx_dual.has_image_2 = True

    ctx_single = SceneContext()
    ctx_single.has_image_1 = True
    ctx_single.has_image_2 = False

    # 1. "What does this image show?" -> GENERIC_IMAGE_DESCRIPTION -> Qwen primary -> ChangeFormer not selected
    p1 = NaturalLanguagePlanner.classify_and_plan("What does this image show?", ctx_single)
    assert p1["intent"] == "GENERIC_IMAGE_DESCRIPTION"
    assert "ChangeFormer" not in p1["relevant_specialists"]
    assert p1["primary_evidence"] == "QWEN2-VL"

    # 2. "Describe the image" -> GENERIC_IMAGE_DESCRIPTION -> Qwen primary
    p2 = NaturalLanguagePlanner.classify_and_plan("Describe the image", ctx_single)
    assert p2["intent"] == "GENERIC_IMAGE_DESCRIPTION"
    assert "ChangeFormer" not in p2["relevant_specialists"]
    assert p2["primary_evidence"] == "QWEN2-VL"

    # 3. "What is in this image?" -> GENERIC_IMAGE_DESCRIPTION -> Qwen primary
    p3 = NaturalLanguagePlanner.classify_and_plan("What is in this image?", ctx_single)
    assert p3["intent"] == "GENERIC_IMAGE_DESCRIPTION"
    assert "ChangeFormer" not in p3["relevant_specialists"]
    assert p3["primary_evidence"] == "QWEN2-VL"

    # 4. "What changed?" -> CHANGE_DETECTION -> ChangeFormer selected
    p4 = NaturalLanguagePlanner.classify_and_plan("What changed?", ctx_dual)
    assert p4["intent"] in ["CHANGE_DETECTION", "CHANGE_ANALYSIS"]
    assert "ChangeFormer" in p4["relevant_specialists"]

    # 5. "How much of the scene changed?" -> CHANGE_DETECTION -> ChangeFormer selected
    p5 = NaturalLanguagePlanner.classify_and_plan("How much of the scene changed?", ctx_dual)
    assert p5["intent"] in ["CHANGE_DETECTION", "CHANGE_ANALYSIS", "CHANGE_MAGNITUDE"]
    assert "ChangeFormer" in p5["relevant_specialists"]

    # 6. "Where are the buildings?" -> OBJECT_DETECTION -> OWL-ViT/Grounding selected
    p6 = NaturalLanguagePlanner.classify_and_plan("Where are the buildings?", ctx_single)
    assert p6["intent"] in ["OBJECT_DETECTION", "OBJECT_LOCATION"]
    assert "OWL-ViT" in p6["relevant_specialists"] or "grounding" in p6["required_tools"]

    # 7. "Analyze the land cover" -> SCENE_UNDERSTANDING -> Qwen primary
    p7 = NaturalLanguagePlanner.classify_and_plan("Analyze the land cover", ctx_single)
    assert p7["intent"] in ["SCENE_UNDERSTANDING", "GENERIC_IMAGE_DESCRIPTION"]
    assert "ChangeFormer" not in p7["relevant_specialists"]
    assert p7["primary_evidence"] == "QWEN2-VL"

    # 8. "Compare these two images" -> CHANGE_DETECTION / IMAGE_COMPARISON -> Qwen + ChangeFormer
    p8 = NaturalLanguagePlanner.classify_and_plan("Compare these two images", ctx_dual)
    assert p8["intent"] in ["CHANGE_DETECTION", "IMAGE_COMPARISON"]
    assert "ChangeFormer" in p8["relevant_specialists"]

    # 9. "Give me a complete report" -> COMPLEX_ANALYSIS -> Qwen + relevant specialists
    p9 = NaturalLanguagePlanner.classify_and_plan("Give me a complete report", ctx_dual)
    assert p9["intent"] in ["COMPLEX_ANALYSIS", "REPORT_REQUEST"]
    assert len(p9["relevant_specialists"]) > 1

    # 10. "Explain the images" with ONE image -> GENERIC_IMAGE_DESCRIPTION
    p10 = NaturalLanguagePlanner.classify_and_plan("Explain the images", ctx_single)
    assert p10["intent"] == "GENERIC_IMAGE_DESCRIPTION"
    assert "ChangeFormer" not in p10["relevant_specialists"]

    # 11. "Explain the images" with TWO images -> IMAGE_COMPARISON / CHANGE_DETECTION
    p11 = NaturalLanguagePlanner.classify_and_plan("Explain the images", ctx_dual)
    assert p11["intent"] in ["IMAGE_COMPARISON", "CHANGE_DETECTION"]
    assert "ChangeFormer" in p11["relevant_specialists"]


def test_missing_input_change_analysis():
    # Only T1 loaded
    ctx_single = SceneContext()
    ctx_single.has_image_1 = True
    ctx_single.has_image_2 = False

    p = NaturalLanguagePlanner.classify_and_plan("What changed?", ctx_single)
    assert p["intent"] in ["CHANGE_DETECTION", "CHANGE_ANALYSIS"]
    assert p["answerability"] == "NOT_ANSWERABLE"
    assert p["missing_input"] == "image_2"
    assert "second temporal" in p.get("unanswerable_reason", "")


def test_unanswerable_sensor_domain_refusal():
    ctx = SceneContext()

    # Temperature
    p_temp = NaturalLanguagePlanner.classify_and_plan("Did temperature increase?", ctx)
    assert p_temp["answerability"] == "NOT_ANSWERABLE"
    assert "thermal" in p_temp["unanswerable_reason"].lower()

    # Rainfall
    p_rain = NaturalLanguagePlanner.classify_and_plan("How much did rainfall increase?", ctx)
    assert p_rain["answerability"] == "NOT_ANSWERABLE"
    assert "meteorological" in p_rain["unanswerable_reason"].lower() or "rainfall" in p_rain["unanswerable_reason"].lower()


def test_ui_action_intents():
    ctx = SceneContext()
    ctx.has_image_1 = True
    ctx.has_image_2 = True

    # Heatmap
    p_hm = NaturalLanguagePlanner.classify_and_plan("show me the heat map", ctx)
    assert p_hm["intent"] == "UI_SHOW_HEATMAP"
    assert p_hm["ui_action"] == {"type": "SELECT_VIEW", "view": "difference-heatmap"}
    assert "navigate_tab:diff-map" in p_hm["ui_actions"]
    assert "ChangeFormer" not in p_hm["relevant_specialists"]

    # Change Mask
    p_cm = NaturalLanguagePlanner.classify_and_plan("show the change mask", ctx)
    assert p_cm["intent"] == "UI_SHOW_CHANGE_MASK"
    assert p_cm["ui_action"] == {"type": "SELECT_VIEW", "view": "change-mask"}
    assert "navigate_tab:change-mask" in p_cm["ui_actions"]

    # Side-by-Side
    p_sbs = NaturalLanguagePlanner.classify_and_plan("show side by side", ctx)
    assert p_sbs["intent"] == "UI_SHOW_SIDE_BY_SIDE"
    assert p_sbs["ui_action"] == {"type": "SELECT_VIEW", "view": "side-by-side"}
    assert "navigate_tab:side-by-side" in p_sbs["ui_actions"]

    # Overlay
    p_ov = NaturalLanguagePlanner.classify_and_plan("show overlay", ctx)
    assert p_ov["intent"] == "UI_SHOW_OVERLAY"
    assert p_ov["ui_action"] == {"type": "SELECT_VIEW", "view": "interactive-overlay"}
    assert "navigate_tab:interactive-overlay" in p_ov["ui_actions"]

    # Reset
    p_rst = NaturalLanguagePlanner.classify_and_plan("reset", ctx)
    assert p_rst["intent"] == "UI_RESET"


def test_object_location_grounding():
    ctx = SceneContext()
    ctx.has_image_1 = True

    p_loc = NaturalLanguagePlanner.classify_and_plan("locate structures", ctx)
    assert p_loc["intent"] in ["OBJECT_LOCATION", "OBJECT_DETECTION"]
    assert "OWL-ViT" in p_loc["relevant_specialists"]
    assert p_loc["grounding_target"] == "structures"

    p_bldg = NaturalLanguagePlanner.classify_and_plan("find buildings", ctx)
    assert p_bldg["intent"] in ["OBJECT_LOCATION", "OBJECT_DETECTION"]
    assert "OWL-ViT" in p_bldg["relevant_specialists"]
    assert p_bldg["grounding_target"] == "buildings"


if __name__ == "__main__":
    test_11_required_routing_scenarios()
    test_missing_input_change_analysis()
    test_unanswerable_sensor_domain_refusal()
    test_ui_action_intents()
    test_object_location_grounding()
    print("[PASS] test_agent_planner.py passed all tests successfully.")

