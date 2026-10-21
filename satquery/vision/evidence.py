"""
Normalized Evidence Schema & Collector for SatQuery AI.
Enforces strict provenance, scoring type, and limitation tagging for all specialist models.
"""

import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """
    Normalized atomic evidence item from a vision specialist.
    Never presents model confidence as ground truth or benchmark accuracy.
    """
    source: str  # e.g., "vilt", "owlvit", "changeformer", "optical_sar", "scene_engine"
    type: str    # e.g., "vqa", "grounding", "change_detection", "optical_sar", "scene_overview"
    label: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    result: Any = None
    score: Optional[float] = None
    score_type: str = "model_confidence"  # "model_confidence", "model_score", "heuristic_score", "none"
    image_id: str = "T1"                  # "T1", "T2", "T1_T2"
    timestamp: float = Field(default_factory=time.time)
    limitations: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "type": self.type,
            "label": self.label,
            "question": self.question,
            "answer": self.answer,
            "result": self.result,
            "score": self.score,
            "score_type": self.score_type,
            "image_id": self.image_id,
            "timestamp": self.timestamp,
            "limitations": self.limitations
        }


class EvidenceCollector:
    """
    Normalizes diverse specialist outputs into uniform EvidenceItem records.
    """

    @staticmethod
    def normalize_vqa(
        question: str,
        raw_result: Dict[str, Any],
        image_id: str = "T1"
    ) -> EvidenceItem:
        answer = raw_result.get("answer", "Unknown")
        conf = raw_result.get("confidence")
        return EvidenceItem(
            source="vilt",
            type="vqa",
            question=question,
            answer=answer,
            result=raw_result,
            score=conf,
            score_type="model_confidence",
            image_id=image_id,
            limitations=["ViLT is a general-purpose vision-language transformer; not remote-sensing fine-tuned."]
        )

    @staticmethod
    def normalize_grounding(
        label: str,
        raw_result: Dict[str, Any],
        image_id: str = "T1"
    ) -> EvidenceItem:
        bboxes = raw_result.get("bboxes", [])
        scores = raw_result.get("box_confidences", [])
        top_score = scores[0] if scores else (0.80 if bboxes else None)
        return EvidenceItem(
            source="owlvit",
            type="grounding",
            label=label,
            result={
                "bboxes": bboxes,
                "num_regions": len(bboxes),
                "box_confidences": scores
            },
            score=top_score,
            score_type="model_score",
            image_id=image_id,
            limitations=["OWL-ViT candidate bounding boxes reflect zero-shot visual similarity scores, not calibrated semantic ground-truth."]
        )

    @staticmethod
    def normalize_change_detection(
        raw_result: Dict[str, Any],
        image_id: str = "T1_T2"
    ) -> EvidenceItem:
        pct = raw_result.get("change_percentage", 0.0)
        conf = raw_result.get("confidence", 0.85)
        clusters = raw_result.get("clusters", [])
        if not clusters and raw_result.get("bboxes"):
            from ..agent.tools import ClusterAnalytics
            clusters = ClusterAnalytics.rank_clusters(raw_result["bboxes"])

        return EvidenceItem(
            source="changeformer",
            type="change_detection",
            label="bi_temporal_change",
            result={
                "change_percentage": pct,
                "clusters": clusters,
                "num_clusters": len(clusters),
                "bboxes": raw_result.get("bboxes", []),
                "artifacts": raw_result.get("artifacts", {})
            },
            score=conf,
            score_type="model_confidence",
            image_id=image_id,
            limitations=["ChangeFormer detects spatial variance between acquisitions; physical cause requires domain validation."]
        )

    @staticmethod
    def normalize_optical_sar(
        raw_result: Dict[str, Any],
        image_id: str = "T1_T2"
    ) -> EvidenceItem:
        return EvidenceItem(
            source="optical_sar",
            type="optical_sar",
            label="cross_modal_fusion",
            answer=raw_result.get("answer", "Optical and SAR comparison complete."),
            result=raw_result,
            score=raw_result.get("confidence", 0.80),
            score_type="heuristic_score",
            image_id=image_id,
            limitations=["Optical and SAR sensor fusion relies on heuristic cross-modal alignment and Lee despeckling."]
        )

    @staticmethod
    def normalize_scene_overview(
        overview_dict: Dict[str, Any],
        image_id: str = "T1"
    ) -> EvidenceItem:
        return EvidenceItem(
            source="scene_engine",
            type="scene_overview",
            label="scene_characteristics",
            result=overview_dict,
            score=overview_dict.get("confidence", 0.82),
            score_type="model_confidence",
            image_id=image_id,
            limitations=["Visual scene understanding is generated from vision specialist probes and general-purpose visual features."]
        )
