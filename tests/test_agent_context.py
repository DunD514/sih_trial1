"""
Lightweight unit tests for SceneContext and Normalized Evidence Layer.
"""

import sys
import os

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.agent.context import SceneContext
from satquery.vision.evidence import EvidenceCollector, EvidenceItem


def test_scene_context_initialization():
    ctx = SceneContext()
    assert isinstance(ctx.images, dict)
    assert isinstance(ctx.modalities, dict)
    assert isinstance(ctx.objects, list)
    assert isinstance(ctx.regions, list)
    assert isinstance(ctx.vqa_results, list)
    assert isinstance(ctx.grounding_results, list)
    assert ctx.change_result is None
    assert ctx.optical_sar_result is None
    assert ctx.active_subject is None
    assert isinstance(ctx.ui_state, dict)
    assert ctx.ui_state.get("active_view") == "overview"


def test_referent_resolution_in_scene_context():
    ctx = SceneContext()
    ctx.change_result = {
        "change_percentage": 20.66,
        "clusters": [
            {"id": "region_1", "index": 0, "name": "Region 1 (Primary)", "area_pixels": 12000, "rank": 1},
            {"id": "region_2", "index": 1, "name": "Region 2", "area_pixels": 4000, "rank": 2}
        ]
    }
    ctx.set_active_cluster(0)

    # 1. "the biggest one"
    ref_biggest = ctx.resolve_referent("the biggest one")
    assert ref_biggest is not None
    assert ref_biggest["id"] == "region_1"

    # 2. "it"
    ref_it = ctx.resolve_referent("it")
    assert ref_it is not None
    assert ref_it["id"] == "region_1"

    # 3. Switch active cluster to Region 2
    ctx.set_active_cluster(1)
    ref_it_2 = ctx.resolve_referent("it")
    assert ref_it_2 is not None
    assert ref_it_2["id"] == "region_2"

    # 4. "this area" / "that region"
    ref_area = ctx.resolve_referent("this area")
    assert ref_area is not None
    assert ref_area["id"] == "region_2"


def test_evidence_collector_normalization():
    # 1. VQA Evidence
    vqa_raw = {"question": "Are there buildings?", "answer": "Yes", "confidence": 0.84}
    ev_vqa = EvidenceCollector.normalize_vqa("Are there buildings?", vqa_raw, image_id="T1")
    assert ev_vqa.source == "vilt"
    assert ev_vqa.type == "vqa"
    assert ev_vqa.score == 0.84
    assert ev_vqa.score_type == "model_confidence"
    assert "not remote-sensing fine-tuned" in ev_vqa.limitations[0]

    # 2. Grounding Evidence
    gnd_raw = {"bboxes": [[100, 80, 200, 180]], "box_confidences": [0.82]}
    ev_gnd = EvidenceCollector.normalize_grounding("buildings", gnd_raw, image_id="T1")
    assert ev_gnd.source == "owlvit"
    assert ev_gnd.type == "grounding"
    assert ev_gnd.score == 0.82
    assert ev_gnd.score_type == "model_score"
    assert "similarity scores" in ev_gnd.limitations[0]

    # 3. Change Detection Evidence
    cd_raw = {"change_percentage": 20.66, "bboxes": [[100, 80, 200, 180]], "confidence": 0.88}
    ev_cd = EvidenceCollector.normalize_change_detection(cd_raw, image_id="T1_T2")
    assert ev_cd.source == "changeformer"
    assert ev_cd.type == "change_detection"
    assert ev_cd.score == 0.88

    # 4. Add to SceneContext
    ctx = SceneContext()
    ctx.add_evidence(ev_vqa.to_dict())
    ctx.add_evidence(ev_gnd.to_dict())
    ctx.add_evidence(ev_cd.to_dict())

    assert len(ctx.vqa_results) == 1
    assert len(ctx.grounding_results) == 1
    assert ctx.change_result is not None
    assert len(ctx.limitations) >= 3


if __name__ == "__main__":
    test_scene_context_initialization()
    test_referent_resolution_in_scene_context()
    test_evidence_collector_normalization()
    print("[PASS] test_agent_context.py passed all tests successfully.")
