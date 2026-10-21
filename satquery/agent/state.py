"""
Centralized Mission State & Agent State Tracking for SatQuery AI.
Tracks multi-modal inputs, analysis history, interactive change clusters,
referent bindings, conversation context, UI states, and JARVIS status.
"""

import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class InputState(BaseModel):
    has_image_1: bool = False
    has_image_2: bool = False
    image_1_path: Optional[str] = None
    image_2_path: Optional[str] = None
    modality_1: str = "optical"
    modality_2: Optional[str] = None
    optical_available: bool = False
    sar_available: bool = False
    is_geotiff_1: bool = False
    is_geotiff_2: bool = False
    dimensions_1: Optional[tuple] = None
    dimensions_2: Optional[tuple] = None


class ChangeState(BaseModel):
    change_percentage: Optional[float] = None
    changed_pixels: Optional[int] = None
    clusters: List[Dict[str, Any]] = Field(default_factory=list)
    selected_cluster_index: Optional[int] = None
    selected_cluster: Optional[Dict[str, Any]] = None
    change_type: Optional[str] = None
    difference_detected: bool = False


class SemanticState(BaseModel):
    vqa_answer: Optional[str] = None
    vqa_top_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    grounding_regions: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_objects: List[str] = Field(default_factory=list)
    scene_description: Optional[str] = None


class CrossModalState(BaseModel):
    optical_evidence: Optional[Dict[str, Any]] = None
    sar_evidence: Optional[Dict[str, Any]] = None
    cross_modal_agreement: Optional[str] = None  # "high", "moderate", "disagree", "unavailable"
    agreement_score: Optional[float] = None
    disagreement_details: Optional[str] = None


class UIState(BaseModel):
    current_tab: str = "overview"
    current_visualization: str = "change-mask"
    selected_cluster_index: Optional[int] = None
    zoom_level: float = 1.0
    active_overlays: List[str] = Field(default_factory=list)
    last_action_executed: Optional[str] = None
    action_status: str = "idle"


class JarvisVoiceState(BaseModel):
    enabled: bool = True
    voice_state: str = "IDLE"  # LISTENING, TRANSCRIBING, THINKING, ANALYZING, NAVIGATING, SPEAKING
    is_speaking: bool = False
    current_utterance: Optional[str] = None


class MissionState(BaseModel):
    """
    Centralized mission and session state for the autonomous satellite analyst copilot.
    """
    session_id: str = "default_session"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # Core sub-states
    inputs: InputState = Field(default_factory=InputState)
    change: ChangeState = Field(default_factory=ChangeState)
    semantic: SemanticState = Field(default_factory=SemanticState)
    cross_modal: CrossModalState = Field(default_factory=CrossModalState)
    ui: UIState = Field(default_factory=UIState)
    jarvis: JarvisVoiceState = Field(default_factory=JarvisVoiceState)

    # Conversational context
    current_query: Optional[str] = None
    last_query: Optional[str] = None
    last_answer: Optional[str] = None
    last_speak_text: Optional[str] = None
    active_subject: Optional[str] = None  # e.g., "Region 1", "buildings", "change_mask"
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    # Structured Decision Support
    why_this_result: Optional[Dict[str, Any]] = None
    why_this_matters: Optional[str] = None
    recommended_next_step: Optional[str] = None
    executive_summary: Optional[Dict[str, Any]] = None
    structured_fusion: Optional[Dict[str, Any]] = None

    # Provenance and Trace
    models_used: List[str] = Field(default_factory=list)
    execution_trace: List[str] = Field(default_factory=list)
    prior_result_id: Optional[str] = None
    report_generated: bool = False

    def log_trace(self, step: str):
        """Append to analyst execution trace."""
        self.execution_trace.append(step)
        self.updated_at = time.time()

    def update_selected_cluster(self, index: int):
        """Sets the selected change cluster and updates referent target."""
        self.change.selected_cluster_index = index
        self.ui.selected_cluster_index = index
        if self.change.clusters and 0 <= index < len(self.change.clusters):
            self.change.selected_cluster = self.change.clusters[index]
            self.active_subject = f"Region {index + 1}"
        self.updated_at = time.time()

    def resolve_referent(self, pronoun_or_phrase: str) -> Optional[Dict[str, Any]]:
        """
        Resolves phrases such as 'it', 'that', 'this area', 'the biggest one', 'the region you found' against mission state.
        """
        phrase = pronoun_or_phrase.lower().strip()
        if any(w in phrase for w in ["biggest", "largest", "most significant", "maximum change", "biggest one"]):
            if self.change.clusters:
                return self.change.clusters[0]
        if phrase in ("it", "that", "this", "there", "the region", "the change", "selected region", "this area", "that area", "the region you found", "the region you identified", "the previous result"):
            if self.change.selected_cluster:
                return self.change.selected_cluster
            elif self.change.clusters:
                return self.change.clusters[0]
        return None


class AgentState:
    """
    Backwards-compatible state class for standalone agent execution pipelines.
    """
    def __init__(self, task_type: str = "auto"):
        self.task_type = task_type
        self.start_time = time.time()
        self.trace: List[str] = []
        self.models_used: List[str] = []
        self.evidence: Dict[str, Any] = {}
        self.warnings: List[str] = []

    def log_step(self, step_name: str, details: Optional[str] = None):
        entry = step_name if details is None else f"{step_name}: {details}"
        self.trace.append(entry)

    def register_model(self, model_name: str):
        if model_name not in self.models_used:
            self.models_used.append(model_name)

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def elapsed_time(self) -> float:
        return round(time.time() - self.start_time, 3)
