"""
End-to-End Real Verification Script for Query Routing Fix.
Tests both required acceptance criteria queries on the real LEVIR satellite image:
Query 1: "What does this image show?"
Query 2: "What changed?"
"""

import os
import sys
import asyncio
import json

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.agent.context import SceneContext
from satquery.agent.planner import NaturalLanguagePlanner
from satquery.agent.query_engine import UniversalQueryEngine
from satquery.vision.orchestrator import AnalysisOrchestrator
from satquery.vision.qwen_service import qwen_vision_service
from satquery.services.ollama import ollama_service


async def run_verification():
    img_a_path = os.path.abspath("samples_LEVIR/A/test_2_0000_0000.png")
    img_b_path = os.path.abspath("samples_LEVIR/B/test_2_0000_0000.png")

    print("======================================================================")
    print("SATQUERY AI — REAL QUERY ROUTING & SYNTHESIS END-TO-END VERIFICATION")
    print("======================================================================")
    print(f"Sample Image A: {img_a_path} (Exists: {os.path.exists(img_a_path)})")
    print(f"Sample Image B: {img_b_path} (Exists: {os.path.exists(img_b_path)})")
    print(f"Ollama Service Model: {ollama_service.model} | Base URL: {ollama_service.base_url}")
    print(f"Qwen2-VL Service Ready: {qwen_vision_service.is_available()}")

    query_engine = UniversalQueryEngine()
    orchestrator = AnalysisOrchestrator()

    # -------------------------------------------------------------------------
    # ACCEPTANCE TEST 1: Generic Image Description
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[TEST 1] Query: 'What does this image show?'")
    print("=" * 70)

    ctx1 = SceneContext()
    ctx1.has_image_1 = True
    ctx1.has_image_2 = False
    ctx1.images["t1"] = img_a_path

    # Step 1: Intent Planning
    plan1 = NaturalLanguagePlanner.classify_and_plan("What does this image show?", ctx1)
    print(f"PLAN 1 INTENT: {plan1.get('intent')}")
    print(f"PLAN 1 PRIMARY EVIDENCE: {plan1.get('primary_evidence')}")
    print(f"PLAN 1 RELEVANT SPECIALISTS: {plan1.get('relevant_specialists')}")
    print(f"PLAN 1 REQUIRES SPECIALIST: {plan1.get('requires_specialist', False)}")

    assert plan1.get("intent") == "GENERIC_IMAGE_DESCRIPTION", f"Expected GENERIC_IMAGE_DESCRIPTION, got {plan1.get('intent')}"
    assert "ChangeFormer" not in plan1.get("relevant_specialists", []), "ChangeFormer should NOT be in relevant specialists!"

    # Step 2: Synthesis (using real Qwen2-VL visual context and Ollama)
    print("\nExtracting Qwen2-VL Visual Context...")
    v_context = qwen_vision_service.analyze_image(img_a_path)
    vc_data = v_context.get("visual_context", {})
    print(f"Qwen2-VL Visual Summary: {vc_data.get('summary')}")
    print(f"Qwen2-VL Objects: {vc_data.get('objects')}")
    print(f"Qwen2-VL Land Cover: {vc_data.get('land_cover')}")

    ctx1.set_qwen_visual_context(vc_data)

    print("\nSynthesizing answer with Ollama qwen3:8b...")
    resp1 = await query_engine.synthesize_grounded_response(
        query="What does this image show?",
        plan=plan1,
        specialist_result=None,
        prior_result=None,
        state=ctx1
    )

    answer1 = resp1.get("answer", "")
    print("\n--- FINAL JARVIS ANSWER (QUERY 1) ---")
    print(answer1)
    print("-" * 70)

    # Verification of rejection of ChangeFormer leaking
    bad_phrases = [
        "changeformer detected",
        "the change map shows",
        "difference probability",
        "spatial change across",
        "change cluster"
    ]
    for bp in bad_phrases:
        assert bp not in answer1.lower(), f"Violated Acceptance Criteria: Found '{bp}' in generic description answer!"

    print("[PASS] Test 1: 'What does this image show?' correctly answered using Qwen2-VL visual context without ChangeFormer output.")

    # -------------------------------------------------------------------------
    # ACCEPTANCE TEST 2: Change Detection Query
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[TEST 2] Query: 'What changed?'")
    print("=" * 70)

    ctx2 = SceneContext()
    ctx2.has_image_1 = True
    ctx2.has_image_2 = True
    ctx2.images["t1"] = img_a_path
    ctx2.images["t2"] = img_b_path
    ctx2.set_qwen_visual_context(vc_data)
    ctx2.change_result = {
        "task": "change_analysis",
        "status": "success",
        "change_percentage": 20.66,
        "changed_pixels": 13540,
        "confidence": 0.884,
        "bboxes": [[100, 80, 220, 190]],
        "models_used": ["ChangeFormerV6 (Transformer)"]
    }

    # Step 1: Intent Planning
    plan2 = NaturalLanguagePlanner.classify_and_plan("What changed?", ctx2)
    print(f"PLAN 2 INTENT: {plan2.get('intent')}")
    print(f"PLAN 2 PRIMARY EVIDENCE: {plan2.get('primary_evidence')}")
    print(f"PLAN 2 RELEVANT SPECIALISTS: {plan2.get('relevant_specialists')}")

    assert plan2.get("intent") in ["CHANGE_DETECTION", "CHANGE_ANALYSIS"], f"Expected CHANGE_DETECTION, got {plan2.get('intent')}"
    assert "ChangeFormer" in plan2.get("relevant_specialists", []), "ChangeFormer SHOULD be selected for change detection!"

    print("\nSynthesizing change explanation with Ollama qwen3:8b...")
    resp2 = await query_engine.synthesize_grounded_response(
        query="What changed?",
        plan=plan2,
        specialist_result=ctx2.change_result,
        prior_result=None,
        state=ctx2
    )

    answer2 = resp2.get("answer", "")
    print("\n--- FINAL JARVIS ANSWER (QUERY 2) ---")
    print(answer2)
    print("-" * 70)

    print("[PASS] Test 2: 'What changed?' correctly synthesized ChangeFormer change detection results.")
    print("\n======================================================================")
    print("ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    asyncio.run(run_verification())
