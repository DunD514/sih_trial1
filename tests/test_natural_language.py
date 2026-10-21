"""
Comprehensive end-to-end multi-turn natural language query & JARVIS conversational tests.
Validates Critical Sequence, Single Image T1 pipeline, and Two Image T1+T2 pipeline.
"""

import sys
import os
import asyncio

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.agent.context import SceneContext
from satquery.agent.planner import NaturalLanguagePlanner
from satquery.agent.query_engine import UniversalQueryEngine
from satquery.agent.tools import UIActionBuilder


class MockFrontendState:
    """Mock frontend state manager."""
    def __init__(self):
        self.active_tab = "overview"
        self.active_view = "overview"
        self.active_overlay = None
        self.selected_cluster = None
        self.evidence_open = False
        self.trace_open = False
        self.zoom_target = None

    def apply_actions(self, actions: list):
        for act in actions:
            name = act.get("action")
            if name == "navigate_tab":
                tab = act.get("tab")
                self.active_tab = tab
                if tab == "diff-map":
                    self.active_view = "heatmap"
                    self.active_overlay = "heatmap"
                elif tab == "change-mask":
                    self.active_view = "change_mask"
                    self.active_overlay = "change_mask"
            elif name in ("show_heatmap", "show_diff_map"):
                self.active_tab = "diff-map"
                self.active_view = "heatmap"
                self.active_overlay = "heatmap"
            elif name == "show_change_mask":
                self.active_tab = "change-mask"
                self.active_view = "change_mask"
            elif name == "show_grounding":
                self.active_overlay = "grounding"
            elif name in ("select_cluster", "highlight_cluster"):
                self.selected_cluster = act.get("index", 0)
            elif name == "zoom_to_cluster":
                self.zoom_target = act.get("index", 0)
            elif name in ("open_evidence", "show_evidence"):
                self.evidence_open = True


def test_critical_multi_turn_sequence():
    """
    CRITICAL TEST (Requirement 34):
    1. 'Give me an analysis on the changes.' -> ChangeFormer evidence.
    2. 'Show me the heatmap.' -> heatmap UI state.
    3. 'What impact could this change have?' -> impact analysis (MUST NOT repeat ChangeFormer result verbatim).
    4. 'What is this image? Give me an overview.' -> visual scene understanding (MUST NOT return ChangeFormer 20.66%).
    """
    qe = UniversalQueryEngine()
    ctx = SceneContext()
    ctx.has_image_1 = True
    ctx.has_image_2 = True
    fe = MockFrontendState()

    mock_cd_result = {
        "task": "change_analysis",
        "change_percentage": 20.66,
        "clusters": [
            {"id": "region_1", "index": 0, "name": "Region 1 (Primary)", "area_pixels": 12000, "rank": 1}
        ],
        "bboxes": [[100, 80, 200, 180]]
    }

    # Step 1: User asks "Give me an analysis on the changes."
    p1 = asyncio.run(qe.plan_investigation("Give me an analysis on the changes.", ctx))
    assert p1["intent"] == "CHANGE_ANALYSIS"
    ctx.change_result = mock_cd_result
    ctx.set_active_cluster(0)

    # Step 2: User asks "Show me the heatmap."
    p2 = asyncio.run(qe.plan_investigation("Show me the heatmap.", ctx, prior_result=mock_cd_result))
    assert p2["intent"] == "UI_COMMAND"
    acts_2 = UIActionBuilder.build_actions(p2, specialist_result=mock_cd_result)
    fe.apply_actions(acts_2)
    assert fe.active_view == "heatmap", f"Expected active_view 'heatmap', got '{fe.active_view}'"

    r2 = asyncio.run(qe.synthesize_grounded_response("Show me the heatmap.", p2, None, mock_cd_result, ctx))
    assert "heatmap" in r2["answer"].lower()
    # Must NOT be an ungrounded raw metrics dump
    assert len(r2["answer"]) < 200

    # Step 3: User asks "What impact could this change have?"
    p3 = asyncio.run(qe.plan_investigation("What impact could this change have?", ctx, prior_result=mock_cd_result))
    assert p3["intent"] == "IMPACT_ANALYSIS"
    r3 = asyncio.run(qe.synthesize_grounded_response("What impact could this change have?", p3, None, mock_cd_result, ctx))
    # Must be impact analysis, NOT simply verbatim ChangeFormer detection statement
    assert any(k in r3["answer"].lower() for k in ["infrastructure", "urban expansion", "monitoring", "cause", "land-use"])

    # Step 4: User asks "What is this image? Give me an overview."
    p4 = asyncio.run(qe.plan_investigation("What is this image? Give me an overview.", ctx, prior_result=mock_cd_result))
    assert p4["intent"] == "SINGLE_IMAGE_OVERVIEW"
    r4 = asyncio.run(qe.synthesize_grounded_response("What is this image? Give me an overview.", p4, None, mock_cd_result, ctx))
    # MUST NOT return "ChangeFormerV6 detected 20.66%..."
    assert "changeformerv6 detected" not in r4["answer"].lower()
    assert "scene" in r4["answer"].lower() or "satellite" in r4["answer"].lower() or "features" in r4["answer"].lower()


def test_single_image_pipeline_t1_only():
    """
    SINGLE IMAGE TEST (Requirement 35):
    With only T1:
    1. 'What is this image?' -> visual understanding.
    2. 'Are there buildings?' -> VQA.
    3. 'Where are they?' -> grounding.
    4. 'Show me them.' -> grounding visualization action.
    5. 'What changed?' -> explain that T2 is required.
    """
    qe = UniversalQueryEngine()
    ctx = SceneContext()
    ctx.has_image_1 = True
    ctx.has_image_2 = False

    # 1. Overview
    p1 = asyncio.run(qe.plan_investigation("What is this image?", ctx))
    assert p1["intent"] == "SINGLE_IMAGE_OVERVIEW"
    r1 = asyncio.run(qe.synthesize_grounded_response("What is this image?", p1, None, None, ctx))
    assert "changeformer" not in r1["answer"].lower()
    assert "scene" in r1["answer"].lower()

    # 2. VQA: Are there buildings?
    p2 = asyncio.run(qe.plan_investigation("Are there buildings?", ctx))
    assert p2["intent"] == "SINGLE_IMAGE_VQA"
    mock_vqa = {"task": "presence_vqa", "answer": "Yes", "confidence": 0.84}
    ctx.add_evidence({"type": "vqa", "result": mock_vqa})

    # 3. Grounding: Where are they?
    p3 = asyncio.run(qe.plan_investigation("Where are they?", ctx))
    assert p3["intent"] == "OBJECT_LOCATION"
    assert p3["grounding_target"] == "buildings"

    # 4. Show me them
    p4 = NaturalLanguagePlanner.classify_and_plan("Show me them.", ctx)
    acts_4 = UIActionBuilder.build_actions(p4)
    assert any(a.get("action") in ("show_grounding", "highlight_cluster", "navigate_tab") for a in acts_4)

    # 5. What changed? (Missing T2 check)
    p5 = asyncio.run(qe.plan_investigation("What changed?", ctx))
    assert p5["intent"] == "CHANGE_ANALYSIS"
    assert p5["missing_input"] == "image_2"
    assert "second" in p5.get("unanswerable_reason", "").lower() or "t2" in p5.get("unanswerable_reason", "").lower()


def test_two_image_pipeline_t1_plus_t2():
    """
    TWO IMAGE TEST (Requirement 36):
    With T1 + T2:
    1. 'What changed?' -> ChangeFormer.
    2. 'Show me the heatmap.' -> UI only.
    3. 'Show me the biggest change.' -> select largest cluster.
    4. 'Zoom into it.' -> zoom selected cluster.
    5. 'Is it near buildings?' -> grounding/spatial reasoning.
    6. 'Why does this matter?' -> impact analysis using existing evidence.
    """
    qe = UniversalQueryEngine()
    ctx = SceneContext()
    ctx.has_image_1 = True
    ctx.has_image_2 = True
    fe = MockFrontendState()

    mock_cd_result = {
        "task": "change_analysis",
        "change_percentage": 20.66,
        "clusters": [
            {"id": "region_1", "index": 0, "name": "Region 1 (Primary)", "area_pixels": 12000, "rank": 1},
            {"id": "region_2", "index": 1, "name": "Region 2", "area_pixels": 3000, "rank": 2}
        ],
        "bboxes": [[100, 80, 200, 180], [30, 40, 70, 90]]
    }

    # 1. "What changed?" -> ChangeFormer
    p1 = asyncio.run(qe.plan_investigation("What changed?", ctx))
    assert p1["intent"] == "CHANGE_ANALYSIS"
    assert p1["requires_specialist"] is True
    ctx.change_result = mock_cd_result

    # 2. "Show me the heatmap." -> UI only
    p2 = asyncio.run(qe.plan_investigation("Show me the heatmap.", ctx, prior_result=mock_cd_result))
    assert p2["intent"] == "UI_COMMAND"
    assert p2["requires_specialist"] is False
    acts_2 = UIActionBuilder.build_actions(p2, mock_cd_result)
    fe.apply_actions(acts_2)
    assert fe.active_view == "heatmap"

    # 3. "Show me the biggest change." -> select largest cluster
    p3 = asyncio.run(qe.plan_investigation("Show me the biggest change.", ctx, prior_result=mock_cd_result))
    assert p3["intent"] == "UI_COMMAND"
    acts_3 = UIActionBuilder.build_actions(p3, mock_cd_result)
    fe.apply_actions(acts_3)
    assert fe.selected_cluster == 0
    ctx.set_active_cluster(0)

    # 4. "Zoom into it." -> zoom selected cluster
    p4 = asyncio.run(qe.plan_investigation("Zoom into it.", ctx, prior_result=mock_cd_result))
    assert p4["intent"] == "UI_COMMAND"
    acts_4 = UIActionBuilder.build_actions(p4, mock_cd_result)
    fe.apply_actions(acts_4)
    assert fe.zoom_target == 0

    # 5. "Is it near buildings?" -> Grounding query
    p5 = asyncio.run(qe.plan_investigation("Is it near buildings?", ctx, prior_result=mock_cd_result))
    assert p5["intent"] in ("OBJECT_LOCATION", "SINGLE_IMAGE_VQA")

    # 6. "Why does this matter?" -> Impact analysis
    p6 = asyncio.run(qe.plan_investigation("Why does this matter?", ctx, prior_result=mock_cd_result))
    assert p6["intent"] == "IMPACT_ANALYSIS"
    assert p6["reuse_prior"] is True


if __name__ == "__main__":
    test_critical_multi_turn_sequence()
    test_single_image_pipeline_t1_only()
    test_two_image_pipeline_t1_plus_t2()
    print("[PASS] test_natural_language.py passed all tests successfully.")
