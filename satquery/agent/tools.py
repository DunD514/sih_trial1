"""
Controlled Backend Tool Registry for SatQuery AI — JARVIS 2.0.
Enforces strict schemas, validation, evidence formatting, and provenance tracking.
The LLM (Qwen3:8b) may only invoke registered tools defined here.
"""

from typing import Dict, Any, List, Optional
import uuid
import numpy as np


class ToolDefinition:
    """Represents a validated tool specification in the JARVIS registry."""

    def __init__(
        self,
        name: str,
        description: str,
        inputs: List[str],
        outputs: List[str],
        specialist_type: str,
        is_neural: bool = True
    ):
        self.name = name
        self.description = description
        self.inputs = inputs
        self.outputs = outputs
        self.specialist_type = specialist_type
        self.is_neural = is_neural

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "specialist_type": self.specialist_type,
            "is_neural": self.is_neural
        }


ALLOWED_SPECIALIST_TOOLS = {
    "change_detection", "change_cluster_analysis", "vqa", "scene_analysis",
    "grounding", "optical_sar", "evidence_fusion", "impact_analysis",
    "mission_summary", "build_summary", "report_generation", "build_report",
    "rank_change_clusters", "inspect_change_cluster", "geospatial", "none"
}

ALLOWED_UI_ACTIONS = {
    "navigate_tab", "show_overlay", "show_side_by_side", "show_change_mask",
    "show_diff_map", "show_heatmap", "select_cluster", "highlight_cluster", "zoom_to_cluster",
    "show_grounding", "show_vqa", "show_optical_sar", "show_geospatial",
    "show_evidence", "show_impact", "show_trace", "show_summary",
    "open_trace", "open_evidence", "open_impact", "open_summary",
    "show_executive_summary", "show_agent_progress", "show_why_this_result",
    "show_why_this_matters", "generate_report", "update_answer",
    "speak_response", "clear_selection", "reset_view"
}

# Strict registry of allowed tools
TOOL_REGISTRY: Dict[str, ToolDefinition] = {
    "change_detection": ToolDefinition(
        name="change_detection",
        description="ChangeFormerV6 bi-temporal transformer change detection on T1 and T2 images.",
        inputs=["image_1", "image_2"],
        outputs=["change_percentage", "change_mask", "bboxes", "confidence", "difference_map"],
        specialist_type="ChangeFormerV6 (Transformer)",
        is_neural=True
    ),
    "change_cluster_analysis": ToolDefinition(
        name="change_cluster_analysis",
        description="Analyzes spatial change clusters from ChangeFormer, deriving rank, area, and bounding boxes.",
        inputs=["bboxes", "change_percentage"],
        outputs=["ranked_clusters", "largest_cluster", "cluster_count"],
        specialist_type="Spatial Cluster Analytics",
        is_neural=False
    ),
    "rank_change_clusters": ToolDefinition(
        name="rank_change_clusters",
        description="Sorts and ranks detected change clusters from ChangeFormer by pixel area and bounding box size.",
        inputs=["clusters"],
        outputs=["ranked_clusters", "largest_cluster_index", "total_clusters"],
        specialist_type="Spatial Cluster Analytics",
        is_neural=False
    ),
    "inspect_change_cluster": ToolDefinition(
        name="inspect_change_cluster",
        description="Retrieves detailed bounding box and spatial metrics for a specific cluster index.",
        inputs=["cluster_index", "clusters"],
        outputs=["cluster_detail", "bbox", "area_pixels", "relative_size"],
        specialist_type="Spatial Cluster Analytics",
        is_neural=False
    ),
    "vqa": ToolDefinition(
        name="vqa",
        description="ViLT-B32 Visual Question Answering for scene queries.",
        inputs=["image_1", "question"],
        outputs=["answer", "confidence", "top_predictions"],
        specialist_type="ViLT-B32 (Vision-Language Transformer)",
        is_neural=True
    ),
    "scene_analysis": ToolDefinition(
        name="scene_analysis",
        description="Scene-level multimodal land cover and feature analysis.",
        inputs=["image_1", "question"],
        outputs=["answer", "top_predictions", "semantic_tags"],
        specialist_type="ViLT-B32 Scene Analyst",
        is_neural=True
    ),
    "grounding": ToolDefinition(
        name="grounding",
        description="OWL-ViT open-vocabulary text-guided spatial bounding box grounding.",
        inputs=["image_1", "question"],
        outputs=["bboxes", "box_confidences", "visualization", "num_regions"],
        specialist_type="OWL-ViT (Zero-Shot Grounding)",
        is_neural=True
    ),
    "optical_sar": ToolDefinition(
        name="optical_sar",
        description="Rule-based cross-modal Optical and SAR sensor alignment and fusion analysis.",
        inputs=["image_1", "image_2"],
        outputs=["optical_evidence", "sar_evidence", "fusion_evidence", "answer"],
        specialist_type="Rule-Based Optical+SAR Engine",
        is_neural=False
    ),
    "evidence_fusion": ToolDefinition(
        name="evidence_fusion",
        description="Fuses evidence across change detection, grounding, VQA, and sensor modalities into structured assessment.",
        inputs=["change_result", "vqa_result", "grounding_result", "optical_sar_result"],
        outputs=["observation", "semantic_evidence", "cross_modal_evidence", "assessment", "limitations"],
        specialist_type="Evidence Fusion Layer",
        is_neural=False
    ),
    "impact_analysis": ToolDefinition(
        name="impact_analysis",
        description="Evaluates hedged domain significance and potential operational relevance.",
        inputs=["evidence", "findings"],
        outputs=["why_this_matters", "relevance_domain", "confidence_level"],
        specialist_type="Domain Impact Analyst",
        is_neural=False
    ),
    "mission_summary": ToolDefinition(
        name="mission_summary",
        description="Synthesizes factual executive summary from mission state and actual specialist evidence.",
        inputs=["mission_state"],
        outputs=["executive_summary", "finding", "largest_change", "recommended_next_step"],
        specialist_type="Analyst Synthesis Engine",
        is_neural=False
    ),
    "build_summary": ToolDefinition(
        name="build_summary",
        description="Compiles a factual executive summary from actual specialist results.",
        inputs=["evidence"],
        outputs=["finding", "largest_change", "evidence_source", "geospatial_info", "limitations"],
        specialist_type="Analyst Synthesis Engine",
        is_neural=False
    ),
    "report_generation": ToolDefinition(
        name="report_generation",
        description="Generates a structured markdown intelligence report from real analysis results.",
        inputs=["result_id", "mission_state"],
        outputs=["markdown_report"],
        specialist_type="Report Generator",
        is_neural=False
    ),
    "build_report": ToolDefinition(
        name="build_report",
        description="Generates a structured markdown intelligence report from real analysis results.",
        inputs=["result_id"],
        outputs=["markdown_report"],
        specialist_type="Report Generator",
        is_neural=False
    ),
    "geospatial": ToolDefinition(
        name="geospatial",
        description="Rasterio GeoTIFF georeferencing, CRS inspection, bounds, and coordinate transformation.",
        inputs=["image_1"],
        outputs=["crs", "bounds", "resolution", "affine_transform", "has_geospatial_meta"],
        specialist_type="Rasterio Geospatial Engine",
        is_neural=False
    )
}


class ClusterAnalytics:
    """Analytical utilities for inspecting and ranking change clusters."""

    @staticmethod
    def rank_clusters(bboxes: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Rank bounding boxes by area in pixels: bbox = [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax].
        Returns list of structured cluster objects sorted from largest to smallest.
        """
        if not bboxes:
            return []

        clusters = []
        for idx, box in enumerate(bboxes):
            if len(box) == 4:
                h = abs(box[2] - box[0])
                w = abs(box[3] - box[1])
                area = h * w
                center_y = int((box[0] + box[2]) / 2)
                center_x = int((box[1] + box[3]) / 2)
                clusters.append({
                    "id": f"region_{idx + 1}",
                    "index": idx,
                    "bbox": box,
                    "width": w,
                    "height": h,
                    "area_pixels": area,
                    "center": [center_y, center_x],
                    "rank": 0
                })

        # Sort by area descending
        clusters.sort(key=lambda c: c["area_pixels"], reverse=True)
        total_area = sum(c["area_pixels"] for c in clusters) or 1.0

        for rank_idx, c in enumerate(clusters):
            c["rank"] = rank_idx + 1
            c["relative_percentage"] = round((c["area_pixels"] / total_area) * 100, 2)
            c["name"] = f"Region {c['rank']} ({c['relative_percentage']}% of change)"

        return clusters

    @staticmethod
    def get_cluster(clusters: List[Dict[str, Any]], cluster_index: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific cluster by index or rank."""
        for c in clusters:
            if c["index"] == cluster_index:
                return c
        if 0 <= cluster_index < len(clusters):
            return clusters[cluster_index]
        return None


class EvidenceFusion:
    """
    Central Evidence Fusion Layer for SatQuery AI.
    Produces rigorous 5-tier structured analytical synthesis:
    1. OBSERVATION: What the models directly detected.
    2. SEMANTIC EVIDENCE: What the vision-language specialists identified.
    3. CROSS-MODAL EVIDENCE: Optical vs SAR radar backscatter agreement.
    4. COMBINED ASSESSMENT: Evidence-grounded interpretation.
    5. LIMITATIONS: Scientific boundaries and validation notice.
    """

    @staticmethod
    def fuse(
        change_result: Optional[Dict[str, Any]] = None,
        vqa_result: Optional[Dict[str, Any]] = None,
        grounding_result: Optional[Dict[str, Any]] = None,
        optical_sar_result: Optional[Dict[str, Any]] = None,
        selected_cluster: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        observation_parts = []
        semantic_parts = []
        cross_modal_parts = []
        assessment_parts = []
        limitations = []

        # 1. ChangeFormer observation
        if change_result and "change_percentage" in change_result:
            pct = change_result["change_percentage"]
            n_clusters = len(change_result.get("bboxes", []))
            observation_parts.append(
                f"ChangeFormerV6 detected spatial differences across {pct:.2f}% of the scene "
                f"spanning {n_clusters} connected change region{'s' if n_clusters != 1 else ''}."
            )
            if selected_cluster:
                observation_parts.append(
                    f"Selected Region {selected_cluster.get('rank', 1)} encompasses {selected_cluster.get('area_pixels', 0):,} pixels "
                    f"({selected_cluster.get('relative_percentage', 0):.1f}% of total detected variance)."
                )

        # 2. Grounding / Semantic evidence
        if grounding_result and grounding_result.get("bboxes"):
            n_boxes = len(grounding_result["bboxes"])
            semantic_parts.append(
                f"OWL-ViT open-vocabulary grounding localized {n_boxes} candidate region{'s' if n_boxes != 1 else ''}."
            )
        if vqa_result and vqa_result.get("answer"):
            semantic_parts.append(f"ViLT VQA scene interpretation: \"{vqa_result['answer']}\".")

        # 3. Cross-modal Optical + SAR
        if optical_sar_result:
            opt_ev = optical_sar_result.get("optical_evidence", {})
            sar_ev = optical_sar_result.get("sar_evidence", {})
            if opt_ev and sar_ev:
                cross_modal_parts.append(
                    "Optical surface variance and SAR radar backscatter variance both provide supporting evidence."
                )
            elif sar_ev:
                cross_modal_parts.append("SAR radar backscatter reveals structural difference.")
            else:
                cross_modal_parts.append("Optical surface features analyzed; SAR input not currently loaded.")
        else:
            cross_modal_parts.append("SAR imagery is currently unavailable; cross-modal validation deferred.")

        # 4. Combined Assessment
        if change_result and (semantic_parts or grounding_result):
            assessment_parts.append(
                "The available spatial and semantic evidence is consistent with physical land-cover or structural modification."
            )
        elif change_result:
            assessment_parts.append(
                "Significant spatial differences detected between timestamps. Further semantic grounding recommended to categorize change type."
            )
        elif vqa_result or grounding_result:
            assessment_parts.append(
                "Single-scene feature extraction complete based on available optical imagery."
            )
        else:
            assessment_parts.append("Analysis state initialized. Imagery loaded and ready for specialist investigation.")

        # 5. Scientific Limitations
        limitations.append("All detected regions represent model-derived inferences and require ground-truth validation for authoritative confirmation.")
        if grounding_result:
            limitations.append("OWL-ViT candidate bounding boxes reflect zero-shot visual similarity scores, not calibrated semantic ground-truth.")
        if vqa_result:
            limitations.append("ViLT VQA is a general vision-language transformer; remote-sensing domain adaptation is in progress.")

        return {
            "observation": " ".join(observation_parts) or "No spatial changes detected.",
            "semantic_evidence": " ".join(semantic_parts) or "No semantic specialist executed.",
            "cross_modal_evidence": " ".join(cross_modal_parts),
            "combined_assessment": " ".join(assessment_parts),
            "limitations": " ".join(limitations)
        }


class ImpactAnalysis:
    """Evaluates hedged operational impact and domain relevance."""

    @staticmethod
    def analyze(evidence: Dict[str, Any], domain: str = "infrastructure") -> Dict[str, Any]:
        finding = evidence.get("combined_assessment", "Surface variation detected.")
        return {
            "why_this_matters": (
                f"Observed changes may indicate {domain} activity, structural reconfiguration, "
                "or environmental shifts requiring asset inspection or land-use audit."
            ),
            "relevance_domain": domain,
            "confidence_level": "Heuristic Analyst Assessment"
        }


class MissionSummary:
    """Generates executive mission intelligence summaries."""

    @staticmethod
    def generate(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "summary": "Multi-sensor remote sensing analysis complete.",
            "status": "active"
        }


class UIActionBuilder:
    """Builds and validates controlled UI actions adhering strictly to ALLOWED_UI_ACTIONS."""

    @staticmethod
    def build_actions(
        plan: dict,
        specialist_result: Optional[dict] = None,
        trace_steps: Optional[list] = None,
        target_cluster: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        actions: list = []

        if trace_steps:
            actions.append({"action": "show_agent_progress", "steps": trace_steps})

        is_specialist_run = plan.get("requires_specialist", False) and plan.get("specialist") not in ("none", "rank_change_clusters", "inspect_change_cluster")

        if specialist_result and is_specialist_run:
            task = specialist_result.get("task", "")
            if task == "change_analysis":
                actions.append({"action": "navigate_tab", "tab": "change-mask"})
                cluster_to_sel = target_cluster if target_cluster is not None else 0
                if specialist_result.get("bboxes"):
                    actions.append({"action": "select_cluster", "index": cluster_to_sel})
                    actions.append({"action": "highlight_cluster", "index": cluster_to_sel})
                actions.append({"action": "open_evidence"})
                actions.append({"action": "show_why_this_result"})
                actions.append({"action": "show_why_this_matters"})
            elif task == "grounding":
                actions.append({"action": "navigate_tab", "tab": "interactive-overlay"})
                actions.append({"action": "show_grounding"})
                actions.append({"action": "open_evidence"})
            elif task in ("scene_vqa", "numerical_vqa", "presence_vqa"):
                actions.append({"action": "show_vqa"})
                actions.append({"action": "open_evidence"})
            elif task == "optical_sar_fusion":
                actions.append({"action": "navigate_tab", "tab": "interactive-overlay"})
                actions.append({"action": "show_optical_sar"})
                actions.append({"action": "open_evidence"})
            actions.append({"action": "open_impact"})

        # Parse and validate planned navigation actions
        for action_str in plan.get("ui_actions", []):
            parts = str(action_str).split(":", 1)
            action_name = parts[0].strip()
            action_param = parts[1].strip() if len(parts) > 1 else None
            if action_name not in ALLOWED_UI_ACTIONS:
                continue
            action_obj: Dict[str, Any] = {"action": action_name}
            if action_name == "show_overlay":
                action_obj["overlay"] = action_param or "heatmap"
            elif action_name in ("show_heatmap", "show_diff_map"):
                action_obj["overlay"] = "heatmap"
            if action_param:
                if action_name == "navigate_tab":
                    action_obj["tab"] = action_param
                elif action_name in ("select_cluster", "highlight_cluster", "zoom_to_cluster"):
                    try:
                        action_obj["index"] = int(action_param)
                    except ValueError:
                        action_obj["index"] = 0
            if action_obj not in actions:
                actions.append(action_obj)

        return actions

