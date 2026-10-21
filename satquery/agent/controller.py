"""
Agent Controller for SatQuery AI: Orchestrates multimodal remote sensing tasks.
Dynamically routes user queries, plans multi-step execution graphs, coordinates specialist models,
derives rigorous multi-modal confidence, and exports structured reports.
"""

from typing import Dict, Any, Union, Optional, List
import numpy as np
from PIL import Image
import os
import threading

from .state import AgentState
from .registry import ModelRegistry
from .planner import TaskPlanner
from .confidence import ConfidenceEngine
from ..preprocessing.validation import InputValidator
from ..preprocessing.geotiff import GeoTIFFProcessor


class AgentController:
    """
    Master multimodal agent controller for SatQuery AI.
    Integrates genuine specialist AI models with dynamic planning, geospatial tracking,
    and calibrated scientific uncertainty estimation.
    """

    def __init__(
        self,
        change_model: Optional[Any] = None,
        vqa: Optional[Any] = None,
        grounding: Optional[Any] = None,
        optical_sar: Optional[Any] = None,
        registry: Optional[ModelRegistry] = None
    ):
        self.registry = registry or ModelRegistry()
        self.planner = TaskPlanner(self.registry)

        # Specialist AI Engines — loaded lazily to allow server startup without checkpoints
        self._change_model_override = change_model
        self._vqa_override = vqa
        self._grounding_override = grounding
        self._optical_sar_override = optical_sar

        self._change_model: Optional[Any] = None
        self._vqa: Optional[Any] = None
        self._grounding: Optional[Any] = None
        self._optical_sar: Optional[Any] = None
        self._change_model_error: Optional[str] = None
        self._vqa_error: Optional[str] = None
        self._grounding_error: Optional[str] = None
        self._optical_sar_error: Optional[str] = None
        self._init_lock = threading.Lock()

        print("===================================================")
        print("SatQuery AI Controller initialised (lazy model loading active).")
        print("Models will be loaded on first inference request.")
        print("===================================================")

    # ------------------------------------------------------------------
    # Lazy model accessors (thread-safe double-checked locking)
    # ------------------------------------------------------------------

    @property
    def change_model(self) -> Optional[Any]:
        if self._change_model_override is not None:
            return self._change_model_override
        if self._change_model is None and self._change_model_error is None:
            with self._init_lock:
                if self._change_model is None and self._change_model_error is None:
                    try:
                        print("[SatQuery] Loading ChangeDetectionModel...")
                        from ..models.change_detection import ChangeDetectionModel
                        self._change_model = ChangeDetectionModel()
                        print("[SatQuery] ChangeDetectionModel loaded.")
                    except Exception as e:
                        self._change_model_error = str(e)
                        print(f"[SatQuery] WARNING: ChangeDetectionModel unavailable — {e}")
        return self._change_model

    @property
    def vqa(self) -> Optional[Any]:
        if self._vqa_override is not None:
            return self._vqa_override
        if self._vqa is None and self._vqa_error is None:
            with self._init_lock:
                if self._vqa is None and self._vqa_error is None:
                    try:
                        print("[SatQuery] Loading RemoteSensingVQA...")
                        from ..models.vqa import RemoteSensingVQA
                        self._vqa = RemoteSensingVQA()
                        print("[SatQuery] RemoteSensingVQA loaded.")
                    except Exception as e:
                        self._vqa_error = str(e)
                        print(f"[SatQuery] WARNING: RemoteSensingVQA unavailable — {e}")
        return self._vqa

    @property
    def grounding(self) -> Optional[Any]:
        if self._grounding_override is not None:
            return self._grounding_override
        if self._grounding is None and self._grounding_error is None:
            with self._init_lock:
                if self._grounding is None and self._grounding_error is None:
                    try:
                        print("[SatQuery] Loading TextGroundingEngine...")
                        from ..models.grounding import TextGroundingEngine
                        self._grounding = TextGroundingEngine()
                        print("[SatQuery] TextGroundingEngine loaded.")
                    except Exception as e:
                        self._grounding_error = str(e)
                        print(f"[SatQuery] WARNING: TextGroundingEngine unavailable — {e}")
        return self._grounding

    @property
    def optical_sar(self) -> Optional[Any]:
        if self._optical_sar_override is not None:
            return self._optical_sar_override
        if self._optical_sar is None and self._optical_sar_error is None:
            with self._init_lock:
                if self._optical_sar is None and self._optical_sar_error is None:
                    try:
                        print("[SatQuery] Loading OpticalSARFusionModel...")
                        from ..models.optical_sar import OpticalSARFusionModel
                        self._optical_sar = OpticalSARFusionModel()
                        print("[SatQuery] OpticalSARFusionModel loaded.")
                    except Exception as e:
                        self._optical_sar_error = str(e)
                        print(f"[SatQuery] WARNING: OpticalSARFusionModel unavailable — {e}")
        return self._optical_sar

    @property
    def qwen_vision(self) -> Any:
        """Dedicated Qwen2-VL-2B-Instruct vision context extraction service."""
        from ..vision.qwen_service import qwen_vision_service
        return qwen_vision_service

    def extract_visual_context(
        self,
        image: Union[str, np.ndarray, Image.Image],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extracts structured 18-field visual context using Qwen2-VL."""
        return self.qwen_vision.analyze_image(image, metadata=metadata)

    def execute(

        self,
        image_1: Union[str, np.ndarray, Image.Image],
        image_2: Optional[Union[str, np.ndarray, Image.Image]] = None,
        question: Optional[str] = None,
        task: Optional[str] = None,
        modality_1: Optional[str] = None,
        modality_2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entrypoint for SatQuery AI.
        """
        state = AgentState()
        state.log_step("input_received")

        # 1. Modality & Geospatial metadata inspection
        mod_1 = InputValidator.infer_modality(image_1, modality_1)
        mod_2 = InputValidator.infer_modality(image_2, modality_2) if image_2 is not None else None
        state.log_step("modality_detected", f"Image 1: {mod_1}" + (f", Image 2: {mod_2}" if mod_2 else ""))

        is_gtiff_1 = isinstance(image_1, str) and GeoTIFFProcessor.is_geotiff(image_1)
        is_gtiff_2 = isinstance(image_2, str) and GeoTIFFProcessor.is_geotiff(image_2)

        geo_meta_1 = GeoTIFFProcessor.inspect(image_1) if is_gtiff_1 else {}
        geo_meta_2 = GeoTIFFProcessor.inspect(image_2) if is_gtiff_2 else {}

        # Check spatial overlap if both are GeoTIFFs
        if is_gtiff_1 and is_gtiff_2:
            overlap_info = GeoTIFFProcessor.check_spatial_overlap(geo_meta_1, geo_meta_2)
            state.log_step("geospatial_overlap_checked", overlap_info.get("overlap_status"))
            if not overlap_info.get("has_overlap", True):
                state.add_warning("Pair images do not share overlapping geographic boundaries.")

        # 2. Plan Execution Steps
        plan_info = self.planner.plan(
            has_image_1=True,
            has_image_2=image_2 is not None,
            modality_1=mod_1,
            modality_2=mod_2,
            question=question,
            explicit_task=task,
            is_geotiff_1=is_gtiff_1,
            is_geotiff_2=is_gtiff_2
        )
        determined_task = plan_info["determined_task"]
        state.log_step("execution_plan_generated", f"Task: {determined_task} ({plan_info['total_stages']} stages)")

        # 3. Route to specialists
        if determined_task in ["change_analysis", "bi_temporal"]:
            if self.change_model is None:
                raise RuntimeError(
                    f"ChangeDetectionModel is not available: {self._change_model_error}. "
                    "Please provide a valid 'best_ckpt.pt' checkpoint in the checkpoints/ directory."
                )
            return self._handle_change_analysis(image_1, image_2, question, state, geo_meta_1, geo_meta_2, plan_info)

        elif determined_task in ["optical_sar_fusion", "optical_sar"]:
            if self.optical_sar is None:
                raise RuntimeError(f"OpticalSARFusionModel is not available: {self._optical_sar_error}")
            return self._handle_optical_sar(image_1, image_2, question, state, mod_1, mod_2, plan_info)

        elif determined_task in ["grounding", "spatial_grounding"]:
            if self.grounding is None:
                raise RuntimeError(f"TextGroundingEngine is not available: {self._grounding_error}")
            return self._handle_grounding(image_1, question, state, geo_meta_1, plan_info)

        elif determined_task == "captioning":
            caption_prompt = question if question else "Describe this remote sensing scene in detail."
            if "describe" not in caption_prompt.lower():
                caption_prompt = "Describe this remote sensing scene in detail. " + caption_prompt
            if self.vqa is None:
                raise RuntimeError(f"RemoteSensingVQA is not available: {self._vqa_error}")
            return self._handle_vqa(image_1, caption_prompt, state, geo_meta_1, plan_info)

        else:  # vqa / scene understanding
            if self.vqa is None:
                raise RuntimeError(f"RemoteSensingVQA is not available: {self._vqa_error}")
            return self._handle_vqa(image_1, question or "What is the primary land cover in this image?", state, geo_meta_1, plan_info)

    def _handle_change_analysis(
        self,
        img_t1: Any,
        img_t2: Any,
        question: Optional[str],
        state: AgentState,
        geo_meta_1: dict,
        geo_meta_2: dict,
        plan_info: dict
    ) -> Dict[str, Any]:
        if img_t2 is None:
            raise ValueError("Bi-temporal change analysis requires both T1 and T2 images.")

        pair_val = InputValidator.validate_pair(img_t1, img_t2, task_type="bi_temporal")
        for w in pair_val.get("warnings", []):
            state.add_warning(w)
        state.log_step("pair_validated")

        # 1. Execute ChangeFormer
        state.register_model(f"{self.change_model.net_G_name} (Executed)")
        state.log_step("changeformer_selected", self.change_model.net_G_name)

        cd_result = self.change_model.predict(img_t1, img_t2)
        state.log_step("changeformer_status: EXECUTED", f"Detected {cd_result['change_percentage']:.2f}% change")

        # 2. Derive Rigorous Scientific Confidence
        conf_data = ConfidenceEngine.derive_change_confidence(
            raw_probabilities=cd_result.get("raw_probabilities"),
            change_mask=cd_result["change_mask"]
        )
        model_conf = conf_data["confidence"]
        state.log_step("confidence_derived", f"Score: {model_conf:.4f} ({conf_data['source']})")

        # 3. GeoJSON polygon generation if georeferenced
        geojson_doc = None
        if geo_meta_1.get("has_geospatial_meta"):
            geojson_doc = GeoTIFFProcessor.export_geojson_report(
                meta=geo_meta_1,
                bboxes=cd_result["bboxes"],
                change_percentage=cd_result["change_percentage"]
            )
            state.log_step("geojson_polygons_exported", f"{len(cd_result['bboxes'])} features")

        num_clusters = len(cd_result["bboxes"])
        change_desc = (
            f"ChangeFormer detected spatial change across {cd_result['change_percentage']:.2f}% of the scene "
            f"({num_clusters} distinct change cluster{'s' if num_clusters != 1 else ''})."
        )
        if question and any(k in question.lower() for k in ["how much", "percentage", "area"]):
            answer = f"The total detected changed area is {cd_result['change_percentage']:.2f}% ({num_clusters} change clusters)."
        else:
            answer = change_desc

        return {
            "task": "change_analysis",
            "question": question or "Describe the changes between T1 and T2",
            "answer": answer,
            "change_type": "Spatial Change Detected",
            "change_percentage": cd_result["change_percentage"],
            "confidence": model_conf,
            "confidence_source": conf_data["source"],
            "uncertainty": conf_data.get("uncertainty", 0.05),
            "specialist_statuses": {
                "ChangeFormerV6": "EXECUTED",
                "GeoTIFFEngine": "EXECUTED" if geo_meta_1.get("has_geospatial_meta") else "READY"
            },
            "change_mask": cd_result["change_mask"],
            "difference_map": cd_result.get("difference_map"),
            "overlay_t1": cd_result["overlay_t1"],
            "overlay_t2": cd_result["overlay_t2"],
            "bboxes": cd_result["bboxes"],
            "geojson": geojson_doc,
            "spatial_evidence": {
                "changed_pixel_count": cd_result["changed_pixels"],
                "total_pixels": cd_result["total_pixels"],
                "num_clusters": num_clusters,
                "top_bboxes": cd_result["bboxes"][:5]
            },
            "geospatial_metadata": {"t1": geo_meta_1, "t2": geo_meta_2},
            "models_used": state.models_used,
            "trace": state.trace,
            "execution_plan": plan_info["execution_plan"],
            "execution_time_sec": state.elapsed_time(),
            "warnings": state.warnings
        }

    def _handle_optical_sar(
        self,
        img_1: Any,
        img_2: Any,
        question: Optional[str],
        state: AgentState,
        mod_1: str,
        mod_2: Optional[str],
        plan_info: dict
    ) -> Dict[str, Any]:
        if img_2 is None:
            raise ValueError("Optical + SAR fusion requires two images.")

        opt_img = img_1 if mod_1 == "optical" else img_2
        sar_img = img_2 if mod_1 == "optical" else img_1

        state.log_step("optical_sar_specialist_selected")
        fusion_res = self.optical_sar.analyze(opt_img, sar_img, question=question)

        for m in fusion_res["models_used"]:
            state.register_model(m)

        state.log_step("optical_sar_fusion_status: EXECUTED")

        return {
            "task": "optical_sar_fusion",
            "question": question or "Synthesize observations from optical and SAR imagery",
            "answer": fusion_res["answer"],
            "confidence": fusion_res["confidence"],
            "confidence_source": fusion_res["confidence_source"],
            "specialist_statuses": {
                "OpticalPreprocessor": "EXECUTED",
                "SARPreprocessor (Lee Filter)": "EXECUTED",
                "OpticalSARFusionEngine": "EXECUTED"
            },
            "optical_evidence": fusion_res["optical_evidence"],
            "sar_evidence": fusion_res["sar_evidence"],
            "fusion_evidence": fusion_res["fusion_evidence"],
            "fusion_visualization": fusion_res["fusion_visualization"],
            "models_used": state.models_used,
            "trace": state.trace,
            "execution_plan": plan_info["execution_plan"],
            "execution_time_sec": state.elapsed_time(),
            "warnings": state.warnings
        }

    def _handle_grounding(
        self,
        img: Any,
        question: Optional[str],
        state: AgentState,
        geo_meta: dict,
        plan_info: dict
    ) -> Dict[str, Any]:
        target_text = question or "built-up structures"
        state.log_step("grounding_specialist_selected", target_text)

        ground_res = self.grounding.ground_text(img, target_text=target_text)
        state.register_model(ground_res["model_name"])
        state.log_step("grounding_status: EXECUTED", f"{ground_res['num_regions']} regions located")

        return {
            "task": "grounding",
            "question": target_text,
            "answer": ground_res["explanation"],
            "confidence": ground_res["confidence"],
            "confidence_source": ground_res["confidence_source"],
            "specialist_statuses": {
                "RSGroundingEngine": "EXECUTED"
            },
            "bboxes": ground_res["bboxes"],
            "box_confidences": ground_res["box_confidences"],
            "num_regions": ground_res["num_regions"],
            "target_mask": ground_res["target_mask"],
            "visualization": ground_res["visualization"],
            "geospatial_metadata": geo_meta,
            "models_used": state.models_used,
            "trace": state.trace,
            "execution_plan": plan_info["execution_plan"],
            "execution_time_sec": state.elapsed_time(),
            "warnings": state.warnings
        }

    def _handle_vqa(
        self,
        img: Any,
        question: str,
        state: AgentState,
        geo_meta: dict,
        plan_info: dict
    ) -> Dict[str, Any]:
        state.log_step("vqa_specialist_selected", question)
        vqa_res = self.vqa.answer(img, question)
        state.register_model(vqa_res["model_name"])
        state.log_step("vqa_status: EXECUTED")

        return {
            "task": vqa_res["task"],
            "question": question,
            "answer": vqa_res["answer"],
            "confidence": vqa_res["confidence"],
            "confidence_source": vqa_res["confidence_source"],
            "scene_classification": vqa_res.get("scene_classification"),
            "top_predictions": vqa_res.get("top_predictions", []),
            "count": vqa_res.get("count"),
            "is_present": vqa_res.get("is_present"),
            "specialist_statuses": {
                "RS-VQA-Engine": "EXECUTED"
            },
            "geospatial_metadata": geo_meta,
            "models_used": state.models_used,
            "trace": state.trace,
            "execution_plan": plan_info["execution_plan"],
            "execution_time_sec": state.elapsed_time(),
            "warnings": state.warnings
        }
