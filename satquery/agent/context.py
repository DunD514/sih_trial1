"""
Centralized Scene Context for SatQuery AI.
Maintains single source of truth for imagery, modalities, scene understanding,
detected objects, spatial regions, VQA results, change results, active referents,
conversation state, UI state, and scientific limitations.
"""

import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SceneContext(BaseModel):
    """
    Unified multimodal scene and mission state context.
    Acts as the single source of truth for the autonomous analyst copilot.
    """
    session_id: str = "default_session"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # Input state & modalities
    images: Dict[str, Any] = Field(default_factory=dict)  # {"t1": path/info, "t2": path/info}
    modalities: Dict[str, str] = Field(default_factory=dict)  # {"t1": "optical", "t2": "optical"}
    has_image_1: bool = False
    has_image_2: bool = False

    # Visual Evidence & Scene Understanding
    scene_understanding: Dict[str, Any] = Field(default_factory=dict)
    objects: List[Dict[str, Any]] = Field(default_factory=list)
    regions: List[Dict[str, Any]] = Field(default_factory=list)
    vqa_results: List[Dict[str, Any]] = Field(default_factory=list)
    grounding_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Multi-temporal & Multi-sensor specialist results
    change_result: Optional[Dict[str, Any]] = None
    optical_sar_result: Optional[Dict[str, Any]] = None

    # Conversational Referent Bindings
    active_subject: Optional[str] = None  # e.g. "scene_overview", "detected_change", "Region 1", "buildings"
    active_region: Optional[Dict[str, Any]] = None
    active_cluster: Optional[Dict[str, Any]] = None
    active_cluster_index: Optional[int] = None
    last_tool_result: Optional[Dict[str, Any]] = None
    last_ui_action: Optional[Dict[str, Any]] = None
    last_answer_subject: Optional[str] = None

    # History & UI State
    conversation_context: List[Dict[str, str]] = Field(default_factory=list)
    ui_state: Dict[str, Any] = Field(default_factory=lambda: {
        "current_tab": "overview",
        "active_view": "overview",
        "selected_cluster": None,
        "zoom_level": 1.0,
        "evidence_open": False,
        "trace_open": False
    })
    available_artifacts: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

    def log_step(self, step: str):
        self.updated_at = time.time()

    def set_active_cluster(self, index: int, cluster_data: Optional[Dict[str, Any]] = None):
        """Sets the active cluster index and referent target."""
        self.active_cluster_index = index
        self.active_subject = f"Region {index + 1}"
        if cluster_data:
            self.active_cluster = cluster_data
            self.active_region = cluster_data
        elif self.change_result and "clusters" in self.change_result:
            clusters = self.change_result["clusters"]
            if 0 <= index < len(clusters):
                self.active_cluster = clusters[index]
                self.active_region = clusters[index]
        self.updated_at = time.time()

    def resolve_referent(self, pronoun_or_phrase: str) -> Optional[Dict[str, Any]]:
        """
        Resolves pronouns ('it', 'that', 'there', 'this area', 'the biggest one', 'the region you found')
        against the current SceneContext.
        """
        phrase = pronoun_or_phrase.lower().strip()

        # Largest / biggest cluster
        if any(w in phrase for w in ["biggest", "largest", "most significant", "maximum change", "biggest one"]):
            if self.active_cluster:
                return self.active_cluster
            if self.change_result and self.change_result.get("clusters"):
                return self.change_result["clusters"][0]
            if self.regions:
                return self.regions[0]

        # Selected / active region or cluster
        if phrase in ("it", "that", "this", "there", "the region", "the change", "selected region", "this area", "that area", "the region you found", "the region you identified", "the previous result"):
            if self.active_cluster:
                return self.active_cluster
            if self.active_region:
                return self.active_region
            if self.change_result and self.change_result.get("clusters"):
                return self.change_result["clusters"][0]
            if self.grounding_results:
                return self.grounding_results[-1]

        # Objects / Buildings
        if any(w in phrase for w in ["buildings", "structures", "vegetation", "roads"]):
            for g in self.grounding_results:
                if any(w in str(g.get("label", "")).lower() for w in ["building", "structure", "vegetation", "road"]):
                    return g

        return None

    def add_evidence(self, evidence_item: Dict[str, Any]):
        """Adds normalized evidence item to context."""
        etype = evidence_item.get("type")
        if etype == "vqa":
            self.vqa_results.append(evidence_item)
        elif etype == "grounding":
            self.grounding_results.append(evidence_item)
            if "bbox" in evidence_item or "regions" in evidence_item:
                self.objects.append(evidence_item)
        elif etype == "scene_overview" or etype == "scene_description":
            self.scene_understanding = evidence_item.get("result", evidence_item)
        elif etype == "change_detection":
            self.change_result = evidence_item.get("result", evidence_item)
            if "clusters" in self.change_result:
                self.regions = self.change_result["clusters"]
        elif etype == "optical_sar":
            self.optical_sar_result = evidence_item.get("result", evidence_item)

        for lim in evidence_item.get("limitations", []):
            if lim not in self.limitations:
                self.limitations.append(lim)

        self.last_tool_result = evidence_item
        self.updated_at = time.time()

    def set_qwen_visual_context(self, visual_context: Dict[str, Any]):
        """Populates scene_understanding and objects from Qwen2-VL structured 20-field context."""
        if not visual_context:
            return
        self.scene_understanding = visual_context
        if "objects" in visual_context and isinstance(visual_context["objects"], list):
            for obj in visual_context["objects"]:
                if isinstance(obj, dict):
                    self.objects.append(obj)
                elif isinstance(obj, str) and obj:
                    self.objects.append({"label": obj, "source": "Qwen2-VL-2B-Instruct"})
        if "uncertainties" in visual_context and isinstance(visual_context["uncertainties"], list):
            for unc in visual_context["uncertainties"]:
                if unc and unc not in self.limitations:
                    self.limitations.append(unc)
        self.updated_at = time.time()

    @property
    def qwen_visual_context(self) -> Optional[Dict[str, Any]]:
        """Returns structured Qwen2-VL visual context if present."""
        return self.scene_understanding if self.scene_understanding else None

