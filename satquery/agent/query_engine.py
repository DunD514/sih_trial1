"""
Universal Satellite Query Engine for SatQuery AI.
Autonomous Multimodal Remote-Sensing Analyst Copilot powered by Qwen3:8b and the Visual Evidence Layer.
Handles question decomposition, referent resolution, controlled tool execution,
anti-hallucination answerability checks, evidence fusion, and dashboard control.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from .context import SceneContext
from .state import MissionState
from .planner import NaturalLanguagePlanner, INTENT_TAXONOMY, ANSWERABILITY_TYPES
from .tools import (
    ALLOWED_SPECIALIST_TOOLS,
    ALLOWED_UI_ACTIONS,
    ClusterAnalytics,
    EvidenceFusion,
    ImpactAnalysis,
    MissionSummary,
    UIActionBuilder
)
from ..vision.scene_understanding import VisualSceneEngine
from ..vision.evidence import EvidenceCollector, EvidenceItem
from ..services.ollama import ollama_service, OLLAMA_MODEL
from ..services.tts import prepare_speech_text

logger = logging.getLogger("satquery.query_engine")


from ..vision.orchestrator import orchestrator
from ..vision.qwen_service import qwen_vision_service


class UniversalQueryEngine:
    """
    Core orchestrator that translates natural language queries into grounded
    multi-step investigations, evidence synthesis, and dashboard actions.
    """

    def __init__(self, ollama=None, vqa_engine=None, grounding_engine=None):
        self.ollama = ollama or ollama_service
        self.vision_engine = VisualSceneEngine()
        self.vqa_engine = vqa_engine
        self.grounding_engine = grounding_engine

    def resolve_context_and_referents(
        self,
        query: str,
        context_or_state: Any
    ) -> Tuple[str, Optional[int], Optional[str]]:
        """
        Resolves pronouns ('it', 'that', 'there', 'the biggest one', 'the region you found', 'this area')
        and binds them to current mission targets.
        Returns (resolved_query_description, target_cluster_index, referent_type).
        """
        q_lower = query.lower().strip()
        selected_idx = 0
        clusters = []

        if isinstance(context_or_state, SceneContext):
            selected_idx = context_or_state.active_cluster_index or 0
            if context_or_state.change_result and "clusters" in context_or_state.change_result:
                clusters = context_or_state.change_result["clusters"]
            selected_cluster = context_or_state.active_cluster
        elif isinstance(context_or_state, MissionState):
            selected_idx = context_or_state.change.selected_cluster_index or 0
            clusters = context_or_state.change.clusters
            selected_cluster = context_or_state.change.selected_cluster
        else:
            selected_cluster = None

        # Referent: "biggest one" / "largest" / "most significant" / "maximum change"
        if any(w in q_lower for w in ["biggest", "largest", "most significant", "rank", "maximum change", "biggest one", "largest area"]):
            return f"{query} [Referent: Largest Change Cluster]", 0, "largest_cluster"

        # Referent: "the region you found" / "the region you identified" / "the previous result"
        if any(w in q_lower for w in ["region you found", "region you identified", "previous result", "the change you found"]):
            if selected_cluster:
                region_name = selected_cluster.get("name", f"Region {selected_idx + 1}")
                return f"{query} [Referent: {region_name}]", selected_idx, "selected_cluster"
            elif clusters:
                return f"{query} [Referent: Region 1 (Primary)]", 0, "primary_cluster"

        # Referent: "that" / "it" / "this" / "there" / "this area" / "that area"
        if re.search(r"\b(it|that|this|there|the region|the change|the area|this area|that area)\b", q_lower):
            if selected_cluster:
                region_name = selected_cluster.get("name", f"Region {selected_idx + 1}")
                return f"{query} [Referent: {region_name}]", selected_idx, "selected_cluster"
            elif clusters:
                return f"{query} [Referent: Region 1 (Primary)]", 0, "primary_cluster"

        return query, selected_idx, None

    def check_answerability(self, query: str, state_or_context: Any) -> Tuple[bool, Optional[str]]:
        """
        Determines whether a question is answerable from remote sensing imagery.
        Refuses queries requiring external non-visual / non-sensor data (e.g. temperature, rainfall, population, subterranean minerals, land ownership).
        Returns (is_answerable, refusal_explanation_if_any).
        """
        q_lower = query.lower()

        # Non-observable thermal / temperature queries (no thermal sensor available)
        if any(w in q_lower for w in ["temperature", "heat index", "thermal", "how hot", "surface temp"]):
            return False, (
                "I can't determine temperature change from the available imagery because no thermal data is available."
            )

        # Non-observable meteorological / precipitation queries
        if any(w in q_lower for w in ["rainfall", "precipitation", "rain", "weather forecast", "humidity"]):
            return False, (
                "I can't determine rainfall or precipitation from the available imagery because no meteorological data is available."
            )

        # Unobservable demographic / non-spectral queries
        if any(w in q_lower for w in ["population", "demographics", "how many people live", "census", "resident count"]):
            return False, (
                "Satellite imagery can identify visible built-up structures and settlement density, "
                "but cannot measure human population directly. An external demographic census dataset is required."
            )

        # Subterranean queries
        if any(w in q_lower for w in ["temperature underground", "subterranean", "deep mineral", "under the ground"]):
            return False, (
                "The available optical and SAR sensors measure surface reflectance and backscatter. "
                "Subterranean or deep subsurface properties cannot be measured from this imagery."
            )

        # Legal / Cadastral ownership queries
        if any(w in q_lower for w in ["who owns", "land ownership", "property title", "cadastral owner"]):
            return False, (
                "Land ownership and parcel titles are legal records and cannot be derived from visual imagery alone."
            )

        return True, None

    def _ensure_scene_context(self, state_or_context: Any) -> SceneContext:
        """Helper to guarantee a valid SceneContext instance."""
        if isinstance(state_or_context, SceneContext):
            return state_or_context

        ctx = SceneContext()
        if isinstance(state_or_context, MissionState):
            ctx.has_image_1 = state_or_context.inputs.has_image_1
            ctx.has_image_2 = state_or_context.inputs.has_image_2
            if state_or_context.inputs.image_1_path:
                ctx.images["t1"] = state_or_context.inputs.image_1_path
            if state_or_context.inputs.image_2_path:
                ctx.images["t2"] = state_or_context.inputs.image_2_path
            if state_or_context.change.change_percentage is not None:
                ctx.change_result = {
                    "change_percentage": state_or_context.change.change_percentage,
                    "clusters": state_or_context.change.clusters
                }
            if state_or_context.change.selected_cluster_index is not None:
                ctx.set_active_cluster(state_or_context.change.selected_cluster_index, state_or_context.change.selected_cluster)
        return ctx

    async def plan_investigation(
        self,
        query: str,
        state_or_context: Any,
        prior_result: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Plans the required specialist tools, question decomposition, and dashboard actions.
        Uses NaturalLanguagePlanner taxonomy and deterministic fallback rules.
        """
        ctx = self._ensure_scene_context(state_or_context)
        if prior_result and not ctx.change_result:
            if prior_result.get("task") == "change_analysis" or "change_percentage" in prior_result:
                ctx.change_result = prior_result

        # 1. Natural Language Plan via Intent Taxonomy
        structured_plan = NaturalLanguagePlanner.classify_and_plan(query, ctx)

        intent = structured_plan.get("intent", "GENERIC_IMAGE_DESCRIPTION")
        specialist_map = {
            "GENERIC_IMAGE_DESCRIPTION": "qwen_vision",
            "SCENE_UNDERSTANDING": "qwen_vision",
            "IMAGE_CAPTIONING": "qwen_vision",
            "SINGLE_IMAGE_OVERVIEW": "qwen_vision",
            "SCENE_DESCRIPTION": "qwen_vision",
            "OBJECT_IDENTIFICATION": "grounding",
            "OBJECT_DETECTION": "grounding",
            "OBJECT_LOCATION": "grounding",
            "SPATIAL_QUERY": "grounding",
            "SINGLE_IMAGE_VQA": "vqa",
            "CHANGE_ANALYSIS": "change_detection" if not structured_plan.get("use_existing_evidence") else "none",
            "CHANGE_DETECTION": "change_detection" if not structured_plan.get("use_existing_evidence") else "none",
            "IMAGE_COMPARISON": "change_detection" if not structured_plan.get("use_existing_evidence") else "none",
            "CHANGE_LOCATION": "inspect_change_cluster",
            "CHANGE_MAGNITUDE": "inspect_change_cluster",
            "CHANGE_INTERPRETATION": "impact_analysis",
            "IMPACT_ANALYSIS": "impact_analysis",
            "SAR_ANALYSIS": "optical_sar",
            "OPTICAL_ANALYSIS": "optical_sar",
            "OPTICAL_SAR_COMPARISON": "optical_sar",
            "EVIDENCE_REQUEST": "evidence_fusion",
            "CONFIDENCE_REQUEST": "evidence_fusion",
            "LIMITATION_REQUEST": "evidence_fusion",
            "RECOMMENDATION": "mission_summary",
            "UI_COMMAND": "none",
            "REPORT_REQUEST": "report_generation",
            "COMPLEX_ANALYSIS": "report_generation"
        }

        specialist = specialist_map.get(intent, "none")
        requires_specialist = structured_plan.get("answerability") in ("REQUIRES_SPECIALIST", "REQUIRES_MULTIPLE_SPECIALISTS")

        legacy_goal_map = {
            "GENERIC_IMAGE_DESCRIPTION": "scene_description",
            "SCENE_UNDERSTANDING": "scene_description",
            "IMAGE_CAPTIONING": "scene_description",
            "SINGLE_IMAGE_OVERVIEW": "single_image_overview",
            "SCENE_DESCRIPTION": "scene_description",
            "OBJECT_IDENTIFICATION": "object_identification",
            "OBJECT_DETECTION": "grounding",
            "OBJECT_LOCATION": "grounding",
            "SPATIAL_QUERY": "grounding",
            "SINGLE_IMAGE_VQA": "vqa",
            "CHANGE_ANALYSIS": "change_detection",
            "CHANGE_DETECTION": "change_detection",
            "IMAGE_COMPARISON": "change_detection",
            "CHANGE_LOCATION": "rank_clusters" if structured_plan.get("subject") == "largest_cluster" else "inspect_location",
            "CHANGE_MAGNITUDE": "inspect_magnitude",
            "CHANGE_INTERPRETATION": "impact",
            "IMPACT_ANALYSIS": "impact",
            "SAR_ANALYSIS": "optical_sar",
            "OPTICAL_ANALYSIS": "optical_sar",
            "OPTICAL_SAR_COMPARISON": "optical_sar",
            "EVIDENCE_REQUEST": "evidence",
            "CONFIDENCE_REQUEST": "evidence",
            "LIMITATION_REQUEST": "limitation_request" if structured_plan.get("answerability") == "NOT_ANSWERABLE" else "evidence",
            "RECOMMENDATION": "investigate_scene",
            "UI_SHOW_HEATMAP": "navigate_heatmap",
            "UI_SHOW_CHANGE_MASK": "navigate_change_mask",
            "UI_SHOW_SIDE_BY_SIDE": "navigate_side_by_side",
            "UI_SHOW_OVERLAY": "navigate_overlay",
            "UI_RESET": "navigate",
            "UI_EXECUTE_ANALYSIS": "execute_analysis",
            "REPORT_REQUEST": "report",
            "COMPLEX_ANALYSIS": "report"
        }

        if intent == "UI_COMMAND":
            if "diff-map" in str(structured_plan.get("ui_actions")):
                legacy_goal = "navigate_heatmap"
            elif "change-mask" in str(structured_plan.get("ui_actions")):
                legacy_goal = "navigate_change_mask"
            elif "zoom_to_cluster" in str(structured_plan.get("ui_actions")):
                legacy_goal = "zoom_cluster"
            elif "show_evidence" in str(structured_plan.get("ui_actions")):
                legacy_goal = "open_evidence"
            elif "show_trace" in str(structured_plan.get("ui_actions")):
                legacy_goal = "open_trace"
            elif "select_cluster" in str(structured_plan.get("ui_actions")):
                legacy_goal = "rank_clusters"
            else:
                legacy_goal = "navigate"
        else:
            legacy_goal = legacy_goal_map.get(intent, "analysis")

        plan_dict = {
            "intent": "UI_COMMAND" if intent.startswith("UI_") else intent,
            "sub_intent": intent,
            "goal": legacy_goal,
            "subject": structured_plan.get("subject", "scene"),
            "requires_specialist": requires_specialist,
            "specialist": specialist,
            "reuse_prior": structured_plan.get("use_existing_evidence", False),
            "missing_input": structured_plan.get("missing_input"),
            "unanswerable_reason": structured_plan.get("unanswerable_reason"),
            "ui_actions": structured_plan.get("ui_actions", []),
            "ui_action": structured_plan.get("ui_action"),
            "response": structured_plan.get("response"),
            "target_cluster": ctx.active_cluster_index or 0,
            "grounding_target": structured_plan.get("grounding_target", "buildings"),
            "response_style": "analytical" if intent != "UI_COMMAND" else "action_confirmation",
            "primary_evidence": structured_plan.get("primary_evidence", "QWEN2-VL"),
            "relevant_specialists": structured_plan.get("relevant_specialists", []),
            "image_count": structured_plan.get("image_count", 1),
            "reason": structured_plan.get("reason", "")
        }

        logger.info("================ QUERY ROUTING ================")
        logger.info("QUERY: %s", query)
        logger.info("INTENT: %s", plan_dict["intent"])
        logger.info("IMAGE_COUNT: %d", plan_dict["image_count"])
        logger.info("PRIMARY EVIDENCE: %s", plan_dict["primary_evidence"])
        logger.info("SPECIALISTS SELECTED: %s", plan_dict["relevant_specialists"])
        logger.info("===============================================")

        return plan_dict

    async def synthesize_grounded_response(
        self,
        query: str,
        plan: dict,
        specialist_result: Optional[dict],
        prior_result: Optional[dict],
        state: Any,
        selected_cluster_info: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes the final analyst response answering the CURRENT question.
        Uses Qwen2-VL visual context as primary evidence for generic image descriptions.
        Applies Hard Response Relevance Guarding to ensure ChangeFormer never pollutes generic visual answers.
        """
        ctx = self._ensure_scene_context(state)
        result_to_use = specialist_result or prior_result or ctx.change_result
        intent = plan.get("intent", "GENERIC_IMAGE_DESCRIPTION")
        goal = plan.get("goal", "")

        # 1. UI Navigation Confirmations (fast sub-second path)
        if intent == "UI_COMMAND" or goal.startswith("navigate_") or goal in ("zoom_cluster", "open_evidence", "open_trace"):
            if goal == "navigate_heatmap":
                return {
                    "answer": "Displaying the difference probability heatmap overlay.",
                    "speak_text": "Displaying difference heatmap.",
                    "why_this_result": {
                        "model": "UI Navigation Controller",
                        "inputs": "Difference Probability Matrix",
                        "output_meaning": "Colormap visualization of spatial change probabilities",
                        "confidence_type": "Deterministic View Switch",
                        "limitations": "Visual probability distribution"
                    },
                    "why_this_matters": "The continuous difference heatmap visualizes subtle spatial gradients across the scene.",
                    "recommended_next_step": "Toggle between the binary mask and continuous difference heatmap to evaluate uncertainty.",
                    "executive_summary": {
                        "finding": "Switched to difference probability heatmap view.",
                        "largest_change": "N/A", "evidence": "Difference heatmap colormap",
                        "geospatial_info": "N/A", "potential_relevance": "Continuous gradient inspection",
                        "limitations": "Visual display"
                    }
                }

            if goal == "navigate_change_mask":
                return {
                    "answer": "Displaying the ChangeFormer binary change mask.",
                    "speak_text": "Displaying change mask.",
                    "why_this_result": {
                        "model": "UI Navigation Controller",
                        "inputs": "ChangeFormer Mask",
                        "output_meaning": "Binary decision mask for significant spatial changes",
                        "confidence_type": "Deterministic View Switch",
                        "limitations": "Thresholded binary mask"
                    },
                    "why_this_matters": "The binary change mask delineates clear boundaries of modified regions.",
                    "recommended_next_step": "Select specific regions to view individual bounding box and area statistics.",
                    "executive_summary": {
                        "finding": "Switched to binary change mask view.",
                        "largest_change": "N/A", "evidence": "Binary mask",
                        "geospatial_info": "N/A", "potential_relevance": "Boundary verification",
                        "limitations": "Visual display"
                    }
                }

            if goal == "zoom_cluster":
                target_idx = plan.get("target_cluster", 0)
                target_name = f"Region {target_idx + 1}"
                if selected_cluster_info and "name" in selected_cluster_info:
                    target_name = selected_cluster_info["name"]
                return {
                    "answer": f"Zoomed into {target_name} on the interactive visual overlay.",
                    "speak_text": f"Zooming into {target_name}.",
                    "why_this_result": {
                        "model": "UI Navigation Controller",
                        "inputs": f"Cluster index {target_idx}",
                        "output_meaning": f"Focused viewport centered on {target_name}",
                        "confidence_type": "Deterministic Action",
                        "limitations": "Viewport focus"
                    },
                    "why_this_matters": "Focused inspection allows verifying fine-grained structural changes and spatial context.",
                    "recommended_next_step": "Run object grounding or scene VQA on this region.",
                    "executive_summary": {
                        "finding": f"Viewport centered on {target_name}.",
                        "largest_change": target_name, "evidence": "Spatial bounding box",
                        "geospatial_info": "Pixel coordinates", "potential_relevance": "Targeted inspection",
                        "limitations": "Viewport focus"
                    }
                }

            if goal in ("open_evidence", "open_trace"):
                confirm_msg = "Opened the agent evidence and execution trace panel." if goal == "open_trace" else "Opened the multi-model evidence panel."
                return {
                    "answer": confirm_msg,
                    "speak_text": "Evidence panel open.",
                    "why_this_result": {
                        "model": "UI Navigation Controller",
                        "inputs": "Evidence and Trace State",
                        "output_meaning": "Audit trail and evidence cards displayed",
                        "confidence_type": "Deterministic Action",
                        "limitations": "System inspection view"
                    },
                    "why_this_matters": "Transparency and provenance verification for AI-generated remote sensing findings.",
                    "recommended_next_step": "Review model outputs and confidence metrics.",
                    "executive_summary": {
                        "finding": confirm_msg,
                        "largest_change": "N/A", "evidence": "Trace and confidence data",
                        "geospatial_info": "N/A", "potential_relevance": "Provenance review",
                        "limitations": "Audit display"
                    }
                }

        # 2. Honest Unanswerable Refusal
        if plan.get("unanswerable_reason"):
            reason = plan["unanswerable_reason"]
            return {
                "answer": reason,
                "speak_text": prepare_speech_text(reason[:120]),
                "why_this_result": {
                    "model": "Answerability Engine",
                    "inputs": "Remote Sensing Imagery",
                    "output_meaning": "Capability boundary identified",
                    "confidence_type": "Domain Boundary Limitation",
                    "limitations": "Requires non-imaging external data"
                },
                "why_this_matters": "Remote sensing analysts must distinguish observable spectral features from external demographic/subsurface data.",
                "recommended_next_step": "Focus inquiries on observable land-cover changes or spatial structures.",
                "executive_summary": {
                    "finding": "Inquiry exceeds optical/SAR sensor capabilities.",
                    "largest_change": "N/A", "evidence": "None",
                    "geospatial_info": "N/A", "potential_relevance": "N/A",
                    "limitations": "Sensor domain limitation"
                }
            }

        # 3. GENERIC IMAGE DESCRIPTION / SCENE UNDERSTANDING (PRIMARY EVIDENCE: QWEN2-VL, NO CHANGEFORMER)
        if intent in ("GENERIC_IMAGE_DESCRIPTION", "SCENE_UNDERSTANDING", "IMAGE_CAPTIONING", "SINGLE_IMAGE_OVERVIEW", "SCENE_DESCRIPTION"):
            img_input = ctx.images.get("t1", "samples_LEVIR/A/test_2_0000_0000.png")
            vc = ctx.qwen_visual_context

            # Extract or retrieve visual context if not cached in SceneContext
            if not vc:
                v_res = orchestrator.extract_visual_context(img_input)
                vc = v_res.get("visual_context")
                if vc:
                    ctx.set_qwen_visual_context(vc)

            # Use compact prompt for generic queries (60% fewer input tokens → faster Ollama)
            fused_prompt = orchestrator.compact_fuse_context_for_ollama(
                base_prompt=query,
                visual_context=vc,
                query_intent=intent,
            )

            # Synthesize answer via Ollama with Hard Response Relevance Guard
            ollama_res = await orchestrator.execute_reasoning_with_ollama(
                fused_prompt=fused_prompt,
                query_intent=intent,
                visual_context=vc
            )

            final_text = (ollama_res.get("analysis") or orchestrator.generate_deterministic_fallback(vc)).strip()
            intro_prefixes = (
                "the satellite scene", "the scene", "the image", "this image",
                "the optical", "this high-resolution", "satellite scene", "a satellite",
                "an aerial", "the aerial"
            )
            if not any(final_text.lower().startswith(p) for p in intro_prefixes) and not any(k in final_text.lower() for k in ["satellite scene", "the scene", "this scene"]):
                final_text = f"The satellite scene shows: {final_text}"
            summary_text = (vc.get("summary") if vc else "Visual scene characteristics") or "Satellite scene overview"

            return {
                "answer": final_text,
                "speak_text": prepare_speech_text(final_text[:120]),
                "why_this_result": {
                    "model": "Qwen2-VL-2B-Instruct + Ollama (qwen3:8b)",
                    "inputs": "Primary Optical Imagery (T1)",
                    "output_meaning": summary_text,
                    "confidence_type": "Multi-Modal Structured Vision Context",
                    "limitations": "Visual observations without ground-truth cadastral overlay"
                },
                "why_this_matters": "Comprehensive scene understanding establishes factual land-cover baseline and spatial distribution.",
                "recommended_next_step": "Ask targeted questions about specific objects (e.g., 'Where are the buildings?') or upload a second image to compare changes.",
                "executive_summary": {
                    "finding": final_text,
                    "largest_change": "N/A",
                    "evidence": summary_text,
                    "geospatial_info": "Standard spatial orientation",
                    "potential_relevance": "Baseline land-cover characterization",
                    "limitations": "Model-derived visual analysis"
                }
            }

        # 4. Impact Analysis (Explaining Significance without repeating raw ChangeFormer metrics)
        if intent == "IMPACT_ANALYSIS" or goal == "impact":
            pct_str = f"{result_to_use['change_percentage']:.2f}%" if (result_to_use and "change_percentage" in result_to_use) else "detected"
            clusters_count = len(result_to_use.get("bboxes", [])) if result_to_use else 1
            impact_text = (
                f"The imagery shows approximately {pct_str} detected spatial difference across {clusters_count} region(s). "
                "That establishes where the observations differ, but it does not establish the definitive cause. "
                "Depending on the underlying features, such changes could be relevant to infrastructure monitoring, "
                "urban expansion assessment, land-use monitoring, or post-event assessment. "
                "The affected regions should be inspected against high-resolution reference data before assigning a specific operational interpretation."
            )
            return {
                "answer": impact_text,
                "speak_text": "The detected spatial variations could indicate urban expansion, infrastructure activity, or land-cover modification.",
                "why_this_result": {
                    "model": "Domain Impact Analyst",
                    "inputs": "Spatial Variance Evidence",
                    "output_meaning": "Domain relevance and potential application mapping",
                    "confidence_type": "Heuristic Domain Interpretation",
                    "limitations": "Interpretation requires authoritative cadastral or ground-truth confirmation"
                },
                "why_this_matters": "Remote sensing analysts must distinguish between raw mathematical pixel differences and operational domain significance.",
                "recommended_next_step": "Select Region 1 on the interactive overlay and execute spatial grounding around affected boundaries.",
                "executive_summary": {
                    "finding": "Spatial variance could indicate infrastructure or land-cover changes.",
                    "largest_change": "Region 1", "evidence": f"{pct_str} spatial difference",
                    "geospatial_info": "Pixel bounding box", "potential_relevance": "Urban & infrastructure monitoring",
                    "limitations": "Model-derived inference; requires ground truth confirmation"
                }
            }

        # 5. Evidence Requests & Limitations
        if intent in ("EVIDENCE_REQUEST", "CONFIDENCE_REQUEST", "LIMITATION_REQUEST"):
            fusion_data = EvidenceFusion.fuse(
                change_result=result_to_use if result_to_use and result_to_use.get("task") == "change_analysis" else None,
                vqa_result=result_to_use if result_to_use and "top_predictions" in result_to_use else None,
                grounding_result=result_to_use if result_to_use and result_to_use.get("task") == "grounding" else None,
                optical_sar_result=result_to_use if result_to_use and "optical_evidence" in result_to_use else None,
                selected_cluster=selected_cluster_info
            )
            ev_answer = (
                f"Supporting evidence: {fusion_data['observation']} "
                f"Semantic evidence: {fusion_data['semantic_evidence']} "
                f"Scientific limitations: {fusion_data['limitations']}"
            )
            return {
                "answer": ev_answer,
                "speak_text": "Evidence is compiled from ChangeFormer spatial difference and vision specialist feature probes.",
                "why_this_result": {
                    "model": "Evidence Fusion Layer",
                    "inputs": "Multi-model Specialist Outputs",
                    "output_meaning": fusion_data["observation"],
                    "confidence_type": "Multi-Modal Calibrated Score",
                    "limitations": fusion_data["limitations"]
                },
                "why_this_matters": "Ensures all AI conclusions are grounded in verifiable sensor measurements rather than hallucinations.",
                "recommended_next_step": "Review model outputs in the Evidence Panel.",
                "executive_summary": {
                    "finding": fusion_data["combined_assessment"],
                    "largest_change": "Region 1", "evidence": fusion_data["observation"],
                    "geospatial_info": "Pixel coordinates", "potential_relevance": "Evidence audit",
                    "limitations": fusion_data["limitations"]
                }
            }

        # 6. Default Fallback Synthesis (5-tier Evidence Fusion)
        fusion_data = EvidenceFusion.fuse(
            change_result=result_to_use if result_to_use and result_to_use.get("task") == "change_analysis" else None,
            vqa_result=result_to_use if result_to_use and "top_predictions" in result_to_use else None,
            grounding_result=result_to_use if result_to_use and result_to_use.get("task") == "grounding" else None,
            optical_sar_result=result_to_use if result_to_use and "optical_evidence" in result_to_use else None,
            selected_cluster=selected_cluster_info
        )

        fallback_ans = (
            result_to_use.get("answer", fusion_data["combined_assessment"]) if result_to_use
            else "Analysis complete based on available satellite imagery."
        )

        return {
            "answer": fallback_ans,
            "speak_text": prepare_speech_text(fallback_ans[:120]),
            "why_this_result": {
                "model": result_to_use.get("task", "ChangeFormerV6") if result_to_use else "SatQuery Analyst",
                "inputs": "Remote Sensing Imagery",
                "output_meaning": fusion_data["observation"],
                "confidence_type": "Model Inference Score",
                "limitations": fusion_data["limitations"]
            },
            "why_this_matters": "Detected spatial patterns could indicate land modification, development, or structural shifts.",
            "recommended_next_step": "Inspect the highest-priority cluster on the interactive visual overlay.",
            "executive_summary": {
                "finding": fallback_ans,
                "largest_change": selected_cluster_info.get("name", "Region 1") if selected_cluster_info else "Region 1",
                "evidence": f"{result_to_use.get('change_percentage', 0):.2f}% spatial change" if (result_to_use and "change_percentage" in result_to_use) else "N/A",
                "geospatial_info": "Standard pixel coordinates",
                "potential_relevance": "Urban & infrastructure monitoring",
                "limitations": fusion_data["limitations"]
            }
        }


# Singleton query engine instance
query_engine = UniversalQueryEngine()

