"""
Visual Evidence Layer and Scene Understanding for SatQuery AI.
Extracts, normalizes, and packages multimodal vision outputs into grounded evidence for Qwen.
"""

from .evidence import EvidenceItem, EvidenceCollector
from .scene_understanding import VisualSceneEngine
from .prompts import SCENE_PROMPTS, OVERVIEW_VQA_QUESTIONS
from .qwen_service import (
    Qwen2VLService,
    qwen_vision_service,
    ImagePreprocessor,
    VisualContextParser,
    EMPTY_VISUAL_CONTEXT,
    QWEN_MODEL_PATH,
    QWEN_DEVICE
)
from .orchestrator import AnalysisOrchestrator, orchestrator

__all__ = [
    "EvidenceItem",
    "EvidenceCollector",
    "VisualSceneEngine",
    "SCENE_PROMPTS",
    "OVERVIEW_VQA_QUESTIONS",
    "Qwen2VLService",
    "qwen_vision_service",
    "ImagePreprocessor",
    "VisualContextParser",
    "EMPTY_VISUAL_CONTEXT",
    "QWEN_MODEL_PATH",
    "QWEN_DEVICE",
    "AnalysisOrchestrator",
    "orchestrator"
]

