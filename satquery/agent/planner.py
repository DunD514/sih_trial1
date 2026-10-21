"""
Intelligent Task Planner, Natural Language Classifier, and Dependency Graph for SatQuery AI.
Implements the 18-intent Remote Sensing Taxonomy, evidence requirements tracking,
and answerability classification.
"""

from typing import Dict, Any, List, Optional, Tuple
import re

from .registry import ModelRegistry
from .context import SceneContext


INTENT_TAXONOMY = {
    "GENERIC_IMAGE_DESCRIPTION",
    "SCENE_UNDERSTANDING",
    "IMAGE_CAPTIONING",
    "SINGLE_IMAGE_OVERVIEW",
    "SCENE_DESCRIPTION",
    "OBJECT_IDENTIFICATION",
    "OBJECT_DETECTION",
    "OBJECT_LOCATION",
    "SPATIAL_QUERY",
    "SINGLE_IMAGE_VQA",
    "CHANGE_ANALYSIS",
    "CHANGE_DETECTION",
    "IMAGE_COMPARISON",
    "CHANGE_LOCATION",
    "CHANGE_MAGNITUDE",
    "CHANGE_INTERPRETATION",
    "IMPACT_ANALYSIS",
    "SAR_ANALYSIS",
    "OPTICAL_ANALYSIS",
    "OPTICAL_SAR_COMPARISON",
    "EVIDENCE_REQUEST",
    "CONFIDENCE_REQUEST",
    "LIMITATION_REQUEST",
    "RECOMMENDATION",
    "UI_COMMAND",
    "REPORT_REQUEST",
    "COMPLEX_ANALYSIS"
}

ANSWERABILITY_TYPES = {
    "DIRECTLY_ANSWERABLE",
    "ANSWERABLE_FROM_EXISTING_EVIDENCE",
    "REQUIRES_SPECIALIST",
    "REQUIRES_MULTIPLE_SPECIALISTS",
    "PARTIALLY_ANSWERABLE",
    "NOT_ANSWERABLE"
}


class NaturalLanguagePlanner:
    """
    Decomposes user queries into structured plans adhering to the strict Intent Taxonomy.
    Determines required evidence, tools, answerability, and UI actions.
    """

    @staticmethod
    def classify_and_plan(
        query: str,
        context: SceneContext
    ) -> Dict[str, Any]:
        """
        Classifies the intent of the current user question against input and conversation state.
        """
        q_lower = query.lower().strip()

        # Compute accurate image cardinality
        image_count = 0
        if context.has_image_1:
            image_count += 1
        if context.has_image_2:
            image_count += 1
        if not image_count and hasattr(context, "images") and context.images:
            image_count = len([p for p in context.images.values() if p])
        if image_count == 0:
            image_count = 1  # default to single loaded image if not specified

        # 1. Fast UI Action Intents (Highest Priority, Direct UI Control)
        # A. Difference Heatmap
        if any(w in q_lower for w in [
            "show me the heat map", "show me the heatmap", "show the heat map", "show the heatmap",
            "display the heat map", "display the heatmap", "bring up the heatmap", "bring up the heat map",
            "show heatmap", "show heat map", "open the heat map", "open the heatmap", "open heat map",
            "open heatmap", "switch to heat map", "switch to heatmap", "view the heat map",
            "view the heatmap", "view heat map", "view heatmap", "show difference heatmap",
            "display difference heatmap", "show diff map", "display diff map", "diff map", "difference map",
        ]) and not any(w in q_lower for w in ["what does the heat map show", "what does the heatmap show", "explain the heat map", "explain the heatmap"]) and not re.search(r"\b(why|how|what|explain)\b", q_lower.replace("show", "")):
            return {
                "intent": "UI_SHOW_HEATMAP",
                "subject": "heatmap_overlay",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:diff-map", "show_heatmap"],
                "ui_action": {"type": "SELECT_VIEW", "view": "difference-heatmap"},
                "response": "Showing the difference heatmap.",
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path UI view switch to difference heatmap"
            }

        # B. Change Mask
        if any(w in q_lower for w in [
            "show the change mask", "show me the change mask", "display the change mask",
            "open change mask", "open the change mask", "show changes on the map",
            "view change mask", "view the change mask", "switch to change mask",
            "show change mask", "display mask", "show mask", "mask view"
        ]):
            return {
                "intent": "UI_SHOW_CHANGE_MASK",
                "subject": "change_mask",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:change-mask", "show_change_mask"],
                "ui_action": {"type": "SELECT_VIEW", "view": "change-mask"},
                "response": "Showing the change mask.",
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path UI view switch to binary change mask"
            }

        # C. Side-by-Side
        if any(w in q_lower for w in [
            "show side by side", "show me side by side", "compare side by side",
            "show both images", "display the images side by side", "display side by side",
            "side by side view", "side by side", "split view"
        ]):
            return {
                "intent": "UI_SHOW_SIDE_BY_SIDE",
                "subject": "side_by_side",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:side-by-side", "show_side_by_side"],
                "ui_action": {"type": "SELECT_VIEW", "view": "side-by-side"},
                "response": "Showing the images side by side.",
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path UI view switch to side-by-side"
            }

        # D. Visual Overlay
        if any(w in q_lower for w in [
            "show overlay", "show the overlay", "show me the overlay",
            "display visual overlay", "display overlay", "visual overlay",
            "overlay view", "switch to overlay", "interactive overlay"
        ]):
            return {
                "intent": "UI_SHOW_OVERLAY",
                "subject": "interactive_overlay",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:interactive-overlay", "show_overlay"],
                "ui_action": {"type": "SELECT_VIEW", "view": "interactive-overlay"},
                "response": "Showing the visual overlay.",
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path UI view switch to visual overlay"
            }

        # E. Reset
        if any(w in q_lower for w in [
            "reset", "clear analysis", "start over", "clear selection", "reset view"
        ]):
            return {
                "intent": "UI_RESET",
                "subject": "reset_dashboard",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:interactive-overlay", "clear_selection", "reset_view"],
                "ui_action": {"type": "SELECT_VIEW", "view": "interactive-overlay"},
                "response": "Resetting the analysis view.",
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path UI reset"
            }

        # F. Execute Analysis
        if any(w in q_lower for w in [
            "run analysis", "execute analysis", "analyze these images", "start analysis", "perform analysis"
        ]):
            return {
                "intent": "UI_EXECUTE_ANALYSIS",
                "subject": "execute_analysis",
                "required_evidence": [],
                "required_tools": ["change_detection"],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["execute_analysis"],
                "ui_action": {"type": "EXECUTE_ANALYSIS"},
                "response": "Executing remote sensing analysis on the loaded imagery.",
                "use_existing_evidence": False,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": ["ChangeFormer"],
                "image_count": image_count,
                "reason": "Fast path execute full analysis"
            }

        if any(w in q_lower for w in ["open evidence", "open the evidence", "show evidence"]):
            return {
                "intent": "UI_COMMAND",
                "subject": "evidence_panel",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["show_evidence", "show_why_this_result"],
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path open evidence panel"
            }

        if any(w in q_lower for w in ["open trace", "open the trace", "show trace", "agent trace", "open the agent trace"]):
            return {
                "intent": "UI_COMMAND",
                "subject": "trace_panel",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["show_trace", "show_agent_progress"],
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path open trace panel"
            }

        if any(w in q_lower for w in ["zoom into the biggest", "zoom into the largest", "zoom into it", "zoom in", "zoom to", "take me to", "focus on"]):
            target_idx = context.active_cluster_index or 0
            return {
                "intent": "UI_COMMAND",
                "subject": f"cluster_{target_idx}",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:interactive-overlay", f"select_cluster:{target_idx}", f"zoom_to_cluster:{target_idx}"],
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path zoom into targeted cluster"
            }

        if any(w in q_lower for w in ["find the biggest change", "which one is biggest", "which is biggest", "find biggest change", "can you find the biggest", "highlight the most significant"]):
            return {
                "intent": "CHANGE_LOCATION",
                "subject": "largest_cluster",
                "required_evidence": ["change_result"],
                "required_tools": ["rank_change_clusters"],
                "answerability": "ANSWERABLE_FROM_EXISTING_EVIDENCE" if context.change_result else "REQUIRES_SPECIALIST",
                "ui_actions": ["navigate_tab:interactive-overlay", "select_cluster:0", "highlight_cluster"],
                "use_existing_evidence": bool(context.change_result),
                "primary_evidence": "ChangeFormer",
                "relevant_specialists": ["ChangeFormer"],
                "image_count": image_count,
                "reason": "Rank and identify largest change cluster"
            }

        if any(w in q_lower for w in ["where is the change", "where did the change occur", "give me the coordinates", "coordinates of change", "change coordinates", "location of change"]):
            return {
                "intent": "CHANGE_LOCATION",
                "subject": "change_location",
                "required_evidence": ["change_result"],
                "required_tools": ["inspect_location"],
                "answerability": "ANSWERABLE_FROM_EXISTING_EVIDENCE" if context.change_result else "REQUIRES_SPECIALIST",
                "ui_actions": ["navigate_tab:interactive-overlay", "highlight_cluster"],
                "use_existing_evidence": bool(context.change_result),
                "primary_evidence": "ChangeFormer",
                "relevant_specialists": ["ChangeFormer"],
                "image_count": image_count,
                "reason": "Identify coordinates and spatial location of change"
            }

        if any(w in q_lower for w in ["show the biggest change", "show me the biggest change", "show me the biggest", "show the largest change", "show me the largest change", "show biggest change", "show largest change", "select largest", "highlight the most significant"]):
            return {
                "intent": "UI_COMMAND",
                "subject": "largest_cluster",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:interactive-overlay", "select_cluster:0", "highlight_cluster"],
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path select and highlight largest change cluster"
            }

        if any(w in q_lower for w in ["visualize the areas", "visualize", "show that", "show the change", "can you visualize"]):
            return {
                "intent": "UI_COMMAND",
                "subject": "interactive_overlay",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:interactive-overlay", "highlight_cluster"],
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path navigate to interactive overlay"
            }

        if any(w in q_lower for w in ["show me them", "show them", "display them"]):
            return {
                "intent": "UI_COMMAND",
                "subject": "grounding_regions",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:interactive-overlay", "show_grounding"],
                "use_existing_evidence": True,
                "primary_evidence": "UI_CONTROLLER",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Fast path display grounding regions"
            }

        # 2. Complex Analysis & Intelligence Reports (Multi-specialist + Qwen)
        if any(w in q_lower for w in [
            "give me a complete analysis", "prepare a report", "analyze everything",
            "detailed remote sensing assessment", "give me a detailed remote sensing assessment",
            "give me a complete report", "generate a report", "export report", "analyst report", "full analysis"
        ]):
            return {
                "intent": "COMPLEX_ANALYSIS",
                "subject": "analyst_report",
                "required_evidence": ["qwen_visual_context", "all_specialist_evidence"],
                "required_tools": ["report_generation"],
                "answerability": "REQUIRES_MULTIPLE_SPECIALISTS",
                "ui_actions": ["generate_report", "show_evidence"],
                "use_existing_evidence": False,
                "primary_evidence": "QWEN2-VL + Specialist Evidence",
                "relevant_specialists": ["ChangeFormer", "OWL-ViT", "ViLT", "Optical_SAR"],
                "image_count": image_count,
                "reason": "Complex multi-modal analyst intelligence report synthesis"
            }

        # 3. Answerability boundary checks (Unobservable sensor features)
        if any(w in q_lower for w in ["temperature", "heat index", "thermal", "how hot", "surface temp"]):
            return {
                "intent": "LIMITATION_REQUEST",
                "subject": "temperature_data",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "NOT_ANSWERABLE",
                "ui_actions": ["show_evidence"],
                "use_existing_evidence": False,
                "unanswerable_reason": "I can't determine temperature change from the available imagery because no thermal infrared sensor data is available.",
                "primary_evidence": "Answerability Engine",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Thermal sensor data missing"
            }

        if any(w in q_lower for w in ["rainfall", "precipitation", "rain", "weather forecast", "humidity"]):
            return {
                "intent": "LIMITATION_REQUEST",
                "subject": "meteorological_data",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "NOT_ANSWERABLE",
                "ui_actions": ["show_evidence"],
                "use_existing_evidence": False,
                "unanswerable_reason": "I can't determine rainfall or precipitation from the available imagery because no meteorological data is available.",
                "primary_evidence": "Answerability Engine",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Meteorological data missing"
            }

        if any(w in q_lower for w in ["population", "demographics", "how many people live", "census", "resident count"]):
            return {
                "intent": "LIMITATION_REQUEST",
                "subject": "demographics",
                "required_evidence": [],
                "required_tools": [],
                "answerability": "NOT_ANSWERABLE",
                "ui_actions": ["show_evidence"],
                "use_existing_evidence": False,
                "unanswerable_reason": "Satellite imagery can identify visible built-up structures and settlement density, but cannot measure human population directly. An external demographic census dataset is required.",
                "primary_evidence": "Answerability Engine",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Demographic census missing"
            }

        # 4. Explicit Change Detection & Image Comparison
        non_change_keywords = [
            "what does this image show", "what does the image show", "describe the image",
            "what is in this image", "explain the image", "significance", "matter",
            "impact", "evidence", "confidence", "why do you", "could you be wrong",
            "coordinates", "where is the change", "where did the change occur", "location of change"
        ]
        is_explicit_change_query = (
            any(w in q_lower for w in [
                "what changed", "what has changed", "show the changes", "show changes", "where has the area changed",
                "explain the difference", "what is different", "how much has the scene changed",
                "how much of the scene changed", "how much changed", "how much change", "detect changes", "detect change",
                "explain the change map", "differences between", "compare these images", "compare the images",
                "compare these two images", "percentage of change", "change percentage",
                "magnitude of change", "area changed", "scene changed"
            ])
            or (("change" in q_lower or "changed" in q_lower) and not any(w in q_lower for w in non_change_keywords))
        )

        # "explain the images" / "describe the images" cardinality rule:
        # If 2+ images -> comparison/change. If 1 image -> generic description.
        is_multi_image_explain = any(w in q_lower for w in ["explain the images", "describe the images", "explain images", "describe images"])
        if is_multi_image_explain and image_count >= 2:
            is_explicit_change_query = True

        if is_explicit_change_query:
            has_cd = bool(context.change_result)
            if "compare" in q_lower or is_multi_image_explain:
                intent_name = "IMAGE_COMPARISON"
            else:
                intent_name = "CHANGE_ANALYSIS"
            if image_count < 2 and not has_cd:
                return {
                    "intent": intent_name,
                    "subject": "temporal_change",
                    "required_evidence": ["change_result"],
                    "required_tools": ["change_detection"],
                    "answerability": "NOT_ANSWERABLE",
                    "ui_actions": ["navigate_tab:overview"],
                    "use_existing_evidence": False,
                    "missing_input": "image_2",
                    "unanswerable_reason": "I have the first image, but bi-temporal change analysis requires a second temporal acquisition.",
                    "primary_evidence": "ChangeFormer",
                    "relevant_specialists": ["ChangeFormer"],
                    "image_count": image_count,
                    "reason": "Change analysis requested without Image 2"
                }

            return {
                "intent": intent_name,
                "subject": "temporal_change",
                "required_evidence": ["change_result", "qwen_visual_context"],
                "required_tools": ["change_detection"] if not has_cd else [],
                "answerability": "ANSWERABLE_FROM_EXISTING_EVIDENCE" if has_cd else "REQUIRES_SPECIALIST",
                "ui_actions": ["navigate_tab:change-mask", "select_cluster:0", "show_evidence"],
                "use_existing_evidence": has_cd,
                "missing_input": None,
                "primary_evidence": "QWEN2-VL + ChangeFormer",
                "relevant_specialists": ["ChangeFormer"],
                "image_count": image_count,
                "reason": "Bi-temporal change detection execution"
            }

        # 5. Generic Image Description & Scene Understanding (QWEN2-VL PRIMARY, NO CHANGEFORMER)
        if any(w in q_lower for w in [
            "what does this image show", "what does the image show", "what do these images show",
            "explain the image", "explain this image", "describe the image", "describe this image",
            "describe scene", "describe this scene", "what is in this image", "what is in the image",
            "what can you see", "tell me about this image", "tell me about the image",
            "what does the scene contain", "what is shown here", "what is shown in this image",
            "analyze this image", "give me an overview", "what is this image", "what am i looking at",
            "tell me what you see", "give me an overview of this image", "overview of the image",
            "explain the images", "describe the images"  # fallback when image_count == 1
        ]):
            intent_name = "SINGLE_IMAGE_OVERVIEW" if ("overview" in q_lower or "what is this image" in q_lower) else "GENERIC_IMAGE_DESCRIPTION"
            return {
                "intent": intent_name,
                "subject": "scene_overview",
                "required_evidence": ["qwen_visual_context"],
                "required_tools": ["qwen_vision"],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:interactive-overlay"],
                "use_existing_evidence": True,
                "primary_evidence": "QWEN2-VL",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "User requested generic visual description of the scene; primary evidence is Qwen2-VL visual context."
            }

        # 6. Scene Understanding / Land Cover / Captioning (QWEN2-VL PRIMARY)
        if any(w in q_lower for w in [
            "give me a caption", "caption this", "summarize the scene", "what kind of area is this",
            "describe the land cover", "analyze the land cover", "land cover", "land use", "what terrain",
            "what vegetation", "what land cover"
        ]):
            return {
                "intent": "SCENE_UNDERSTANDING",
                "subject": "land_cover_and_use",
                "required_evidence": ["qwen_visual_context"],
                "required_tools": ["qwen_vision"],
                "answerability": "DIRECTLY_ANSWERABLE",
                "ui_actions": ["navigate_tab:interactive-overlay"],
                "use_existing_evidence": True,
                "primary_evidence": "QWEN2-VL",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "User queried semantic land cover, vegetation, or scene characteristics."
            }

        # 7. Object Detection & Spatial Grounding (OWL-ViT Specialist)
        if any(w in q_lower for w in [
            "locate structures", "locate the structures", "find buildings", "find structures",
            "where are the buildings", "where are buildings", "where are the structures",
            "where are structures", "are there vehicles", "where are the roads", "detect ships",
            "find roads", "find houses", "detect buildings", "detect structures", "locate the",
            "locate", "where is the", "where are the", "built-up region", "where are",
            "what objects are visible", "list objects", "what features are visible"
        ]):
            target = "structures" if "structure" in q_lower else ("buildings" if any(w in q_lower for w in ["building", "house", "built-up", "they"]) else ("roads" if "road" in q_lower else "structures"))
            return {
                "intent": "OBJECT_LOCATION",
                "subject": target,
                "required_evidence": ["grounding_regions", "qwen_visual_context"],
                "required_tools": ["grounding"],
                "grounding_target": target,
                "answerability": "REQUIRES_SPECIALIST",
                "ui_actions": ["navigate_tab:interactive-overlay", "show_grounding", "show_evidence"],
                "use_existing_evidence": False,
                "primary_evidence": "OWL-ViT Grounding",
                "relevant_specialists": ["OWL-ViT"],
                "image_count": image_count,
                "reason": f"Spatial object detection and grounding for '{target}'"
            }

        # 8. Optical vs SAR Cross-Modal Analysis
        if any(w in q_lower for w in ["analyze the sar image", "what does the radar image show", "identify sar characteristics", "compare the optical and sar", "compare optical and sar", "radar comparison", "sar"]):
            return {
                "intent": "SAR_ANALYSIS",
                "subject": "cross_modal_sensor_fusion",
                "required_evidence": ["optical_sar_evidence", "qwen_visual_context"],
                "required_tools": ["optical_sar"],
                "answerability": "REQUIRES_SPECIALIST",
                "ui_actions": ["navigate_tab:interactive-overlay", "show_optical_sar", "show_evidence"],
                "use_existing_evidence": False,
                "primary_evidence": "SAR Specialist + QWEN2-VL",
                "relevant_specialists": ["SAR"],
                "image_count": image_count,
                "reason": "SAR radar backscatter and cross-modal analysis"
            }

        if any(w in q_lower for w in ["analyze the optical image", "what are the optical characteristics", "optical characteristics", "optical analysis"]):
            return {
                "intent": "OPTICAL_ANALYSIS",
                "subject": "optical_spectral_characteristics",
                "required_evidence": ["optical_evidence", "qwen_visual_context"],
                "required_tools": ["optical_sar"],
                "answerability": "REQUIRES_SPECIALIST",
                "ui_actions": ["navigate_tab:interactive-overlay", "show_optical_sar", "show_evidence"],
                "use_existing_evidence": False,
                "primary_evidence": "Optical Specialist + QWEN2-VL",
                "relevant_specialists": ["Optical"],
                "image_count": image_count,
                "reason": "Optical spectral feature analysis"
            }

        # 9. Targeted Single-Image VQA
        if any(w in q_lower for w in ["are there", "is there", "is it", "how many", "count", "is that a"]):
            return {
                "intent": "SINGLE_IMAGE_VQA",
                "subject": "visual_query",
                "required_evidence": ["vqa_prediction", "qwen_visual_context"],
                "required_tools": ["vqa"],
                "answerability": "REQUIRES_SPECIALIST",
                "ui_actions": ["show_vqa", "show_evidence"],
                "use_existing_evidence": False,
                "primary_evidence": "ViLT VQA + QWEN2-VL",
                "relevant_specialists": ["ViLT"],
                "image_count": image_count,
                "reason": f"Targeted visual question answering for '{query}'"
            }

        # 10. Impact Analysis & Domain Significance
        if any(w in q_lower for w in ["why does this matter", "what impact could this change have", "what impact could this have", "what could this change indicate", "why is this significant", "significance of this change"]):
            has_cd = bool(context.change_result)
            return {
                "intent": "IMPACT_ANALYSIS",
                "subject": "current_change_impact",
                "required_evidence": ["change_result"] if has_cd else [],
                "required_tools": ["impact_analysis"] if not has_cd else [],
                "answerability": "ANSWERABLE_FROM_EXISTING_EVIDENCE" if has_cd else "DIRECTLY_ANSWERABLE",
                "ui_actions": ["show_why_this_matters", "open_impact"],
                "use_existing_evidence": True,
                "primary_evidence": "Impact Analysis Engine",
                "relevant_specialists": ["ChangeFormer"] if has_cd else [],
                "image_count": image_count,
                "reason": "Domain significance and operational relevance analysis"
            }

        # 11. Evidence Requests, Confidence, Limitations & Next Steps
        if any(w in q_lower for w in ["why do you think that", "what evidence supports that", "evidence for this", "show me the evidence"]):
            return {
                "intent": "EVIDENCE_REQUEST",
                "subject": "evidence_chain",
                "required_evidence": ["evidence_fusion"],
                "required_tools": [],
                "answerability": "ANSWERABLE_FROM_EXISTING_EVIDENCE",
                "ui_actions": ["show_why_this_result", "show_evidence"],
                "use_existing_evidence": True,
                "primary_evidence": "Evidence Fusion Layer",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "User requested the supporting evidence chain"
            }

        if any(w in q_lower for w in ["how confident are you", "confidence score", "confidence level"]):
            return {
                "intent": "CONFIDENCE_REQUEST",
                "subject": "model_confidence",
                "required_evidence": ["confidence_metrics"],
                "required_tools": [],
                "answerability": "ANSWERABLE_FROM_EXISTING_EVIDENCE",
                "ui_actions": ["show_why_this_result", "show_evidence"],
                "use_existing_evidence": True,
                "primary_evidence": "Confidence Engine",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "User queried confidence derivation and metrics"
            }

        if any(w in q_lower for w in ["could you be wrong", "limitations", "what are the limitations", "uncertainty"]):
            return {
                "intent": "LIMITATION_REQUEST",
                "subject": "scientific_limitations",
                "required_evidence": ["limitations"],
                "required_tools": [],
                "answerability": "ANSWERABLE_FROM_EXISTING_EVIDENCE",
                "ui_actions": ["show_why_this_result", "show_evidence"],
                "use_existing_evidence": True,
                "primary_evidence": "Limitation Engine",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "User queried potential errors, domain limits, and scientific boundaries"
            }

        if any(w in q_lower for w in ["what should i investigate next", "what should i investigate", "what next", "recommended next step", "give me a summary", "summary", "summarize", "investigate", "pay attention"]):
            return {
                "intent": "RECOMMENDATION",
                "subject": "next_steps",
                "required_evidence": ["mission_summary"],
                "required_tools": [],
                "answerability": "ANSWERABLE_FROM_EXISTING_EVIDENCE",
                "ui_actions": ["show_executive_summary", "show_evidence"],
                "use_existing_evidence": True,
                "primary_evidence": "Mission Summary Engine",
                "relevant_specialists": [],
                "image_count": image_count,
                "reason": "Actionable next steps and scene summary recommendation"
            }

        # Default fallback: Treat as generic image description if not matching
        return {
            "intent": "GENERIC_IMAGE_DESCRIPTION",
            "subject": "general_scene_inquiry",
            "required_evidence": ["qwen_visual_context"],
            "required_tools": ["qwen_vision"],
            "answerability": "DIRECTLY_ANSWERABLE",
            "ui_actions": ["navigate_tab:interactive-overlay"],
            "use_existing_evidence": True,
            "primary_evidence": "QWEN2-VL",
            "relevant_specialists": [],
            "image_count": image_count,
            "reason": "Conversational remote sensing scene inquiry"
        }


class TaskPlanner:
    """
    Backwards-compatible execution graph sequencer.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    def plan(
        self,
        has_image_1: bool,
        has_image_2: bool,
        modality_1: str,
        modality_2: Optional[str],
        question: Optional[str],
        explicit_task: Optional[str],
        is_geotiff_1: bool,
        is_geotiff_2: bool
    ) -> Dict[str, Any]:
        q_lower = (question or "").lower().strip()
        steps: List[Dict[str, Any]] = []

        steps.append({
            "stage": 1,
            "name": "Input & Modality Validation",
            "action": "validate_inputs",
            "description": f"Validate Image 1 ({modality_1})" + (f" and Image 2 ({modality_2})" if has_image_2 else "")
        })

        if explicit_task:
            task_type = explicit_task.lower()
        elif has_image_2:
            if (modality_1 == "optical" and modality_2 == "sar") or "sar" in q_lower:
                task_type = "optical_sar_fusion"
            else:
                task_type = "change_analysis"
        elif any(k in q_lower for k in ["where", "locate", "ground", "find the", "highlight", "bounding box"]):
            task_type = "grounding"
        elif any(k in q_lower for k in ["overview", "what is this image", "tell me what you see"]):
            task_type = "captioning"
        else:
            task_type = "vqa"

        return {
            "determined_task": task_type,
            "total_stages": len(steps) + 1,
            "execution_plan": steps
        }
