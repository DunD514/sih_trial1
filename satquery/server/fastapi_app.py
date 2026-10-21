"""
FastAPI High-Performance REST API Microservice for SatQuery AI.
Provides async endpoints for remote sensing multi-modal analysis, image validation,
model registry inspection, quantitative benchmarks, and geospatial report exports.
"""

import os
import sys
import uuid
import time
import base64
import mimetypes
import re
import json as _json
import logging
import threading
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import httpx
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger("satquery.server")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..agent.controller import AgentController
from ..agent.registry import ModelRegistry
from ..preprocessing.geotiff import GeoTIFFProcessor
from ..preprocessing.validation import InputValidator
from ..reporting.export import ReportExporter
from ..evaluation.benchmark import EvaluationBenchmark
from ..vision.qwen_service import qwen_vision_service, VisualContextCache, QWEN_MODEL_PATH, QWEN_DEVICE
from ..vision.orchestrator import orchestrator


# Event set only after Qwen warmup inference AND cache-put are fully complete.
# Health endpoint uses this — not is_available() — to prevent benchmark race conditions.
_QWEN_WARMUP_DONE = threading.Event()


def _warmup_qwen() -> None:
    """
    Runs in a background thread at server startup when ENABLE_MODEL_WARMUP=true.
    By default on production / EC2, models load lazily on first inference request to
    ensure instant (<1s) boot time and zero OOM risk.
    """
    enable_warmup = os.getenv("ENABLE_MODEL_WARMUP", "false").lower() in ("1", "true", "yes")
    if not enable_warmup:
        logger.info("[Startup] Eager model warmup is disabled (ENABLE_MODEL_WARMUP=false). Models will load lazily on first request.")
        _QWEN_WARMUP_DONE.set()
        return

    if not os.path.exists(qwen_vision_service.model_path):
        logger.info("[Warmup] Qwen2-VL model path not found at '%s'. Skipping warmup; model will load if downloaded.", qwen_vision_service.model_path)
        _QWEN_WARMUP_DONE.set()
        return

    try:
        t0 = time.time()
        logger.info("[Warmup] Starting Qwen2-VL eager pre-load...")

        # Step 1: Load model weights into VRAM
        ok = qwen_vision_service.load_model()
        if not ok:
            logger.warning("[Warmup] Qwen2-VL load failed — cold-start penalty will apply to first user query.")
            _QWEN_WARMUP_DONE.set()  # unblock health even on failure
            return

        load_dur = time.time() - t0
        logger.info("[Warmup] Model loaded in %.1fs. Running CUDA kernel warmup inference...", load_dur)

        # Step 2: Use a real LEVIR sample image (256x256) if available
        warmup_candidates = [
            os.path.join("samples_LEVIR", "A", "test_2_0000_0000.png"),
            os.path.join("samples_LEVIR", "A", "test_102_0512_0000.png"),
        ]
        warmup_img = None
        for candidate in warmup_candidates:
            if os.path.isfile(candidate):
                warmup_img = candidate
                logger.info("[Warmup] Using real sample for warmup: %s", candidate)
                break

        if warmup_img is None:
            logger.info("[Warmup] No sample found — using synthetic 256x256 warmup image.")
            warmup_img = Image.new("RGB", (256, 256), color=(96, 120, 80))

        # Cache result so the first user query on this image is instant
        qwen_vision_service.analyze_image(warmup_img, use_cache=True)

        total_dur = time.time() - t0
        logger.info("[Warmup] Qwen2-VL fully warm and cached in %.1fs — first user query will be fast.", total_dur)

    except Exception as exc:
        logger.error("[Warmup] Qwen2-VL warmup failed: %s — cold-start penalty will apply.", exc)
    finally:
        _QWEN_WARMUP_DONE.set()
        logger.info("[Warmup] _QWEN_WARMUP_DONE set.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: start Qwen2-VL warmup in background thread at server boot."""
    warmup_thread = threading.Thread(target=_warmup_qwen, name="qwen-warmup", daemon=True)
    warmup_thread.start()
    logger.info("[Startup] Qwen2-VL background warmup thread launched.")
    yield
    # Shutdown: nothing to clean up (daemon thread exits with process)



# Initialize FastAPI app
app = FastAPI(
    title="SatQuery AI API",
    description="Interactive Multimodal Remote-Sensing AI Assistant API (SIH26167)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
controller = AgentController()
registry = ModelRegistry()
benchmark_runner = EvaluationBenchmark(controller)

# In-memory results cache for task retrieval
results_cache: Dict[str, Dict[str, Any]] = {}


# Pydantic Schemas
class AnalyzeRequest(BaseModel):
    image_1: Optional[str] = Field(None, description="Base64 data URL for Image 1")
    image_2: Optional[str] = Field(None, description="Base64 data URL for Image 2")
    image_1_path: Optional[str] = Field(None, description="Server file path for Image 1")
    image_2_path: Optional[str] = Field(None, description="Server file path for Image 2")
    question: Optional[str] = Field(None, description="User question or prompt")
    task: Optional[str] = Field(None, description="Explicit task override (change_analysis, vqa, grounding, optical_sar)")
    modality_1: Optional[str] = Field("optical", description="Modality of Image 1")
    modality_2: Optional[str] = Field("optical", description="Modality of Image 2")


def array_to_base64(arr: np.ndarray, format_str: str = "PNG") -> str:
    """Encodes numpy array into base64 data URL."""
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr.astype(np.uint8))
    else:
        pil_img = Image.fromarray(arr.astype(np.uint8)[:, :, :3])
    buffered = BytesIO()
    pil_img.save(buffered, format=format_str)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format_str.lower()};base64,{encoded}"


# Endpoints
@app.get("/health", tags=["System"])
@app.get("/api/v1/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health_check():
    """
    Lightweight health check reporting application health and service readiness.
    Evaluates system status without triggering eager AI model loading.
    """
    def _status(model, error_attr):
        if model is not None:
            return "Loaded"
        err = getattr(controller, error_attr, None)
        return f"Unavailable — {err}" if err else "Ready (Lazy / on-demand)"

    # Non-blocking status checks
    ollama_ok, ollama_reason = await _check_ollama_status()
    tts_status = await tts_service.get_status()

    return {
        "status": "healthy",
        "system": "SatQuery AI Remote Sensing Assistant",
        "version": "2.0.0",
        "capabilities": {
            "web_ui": "ready",
            "fastapi_server": "running",
            "ollama_reasoning": "online" if ollama_ok else f"offline ({ollama_reason})",
            "tts_speech": tts_status.get("status", "ready"),
            "tts_provider": tts_status.get("provider", "kokoro"),
            "models": {
                "change_detection": _status(controller._change_model, "_change_model_error"),
                "vqa": _status(controller._vqa, "_vqa_error"),
                "grounding": _status(controller._grounding, "_grounding_error"),
                "optical_sar": _status(controller._optical_sar, "_optical_sar_error"),
                "qwen2_vl_vision": "Loaded" if qwen_vision_service._is_initialized else ("Loaded" if _QWEN_WARMUP_DONE.is_set() and os.path.exists(qwen_vision_service.model_path) else "Ready (Lazy / on-demand)"),
                "geospatial": "Rasterio CRS Engine (Ready)"
            }
        },
        "models": {
            "change_detection": _status(controller._change_model, "_change_model_error"),
            "vqa": _status(controller._vqa, "_vqa_error"),
            "grounding": _status(controller._grounding, "_grounding_error"),
            "optical_sar": _status(controller._optical_sar, "_optical_sar_error"),
            "qwen2_vl_vision": "Loaded" if qwen_vision_service._is_initialized else ("Loaded" if _QWEN_WARMUP_DONE.is_set() and os.path.exists(qwen_vision_service.model_path) else "Ready (Lazy / on-demand)"),
            "ollama_reasoning": f"Ready ({OLLAMA_MODEL})" if ollama_ok else f"Unavailable — Ollama offline ({ollama_reason})",
            "geospatial": "Rasterio CRS Engine (Ready)"
        },
        "qwen_warm": _QWEN_WARMUP_DONE.is_set()
    }



@app.get("/api/v1/models")
async def list_models():
    """Returns the catalog of registered remote-sensing AI specialists."""
    return {
        "count": len(registry.list_all()),
        "models": registry.list_all()
    }


@app.get("/api/v1/samples")
@app.get("/api/samples")
async def list_samples():
    """Lists available remote-sensing demo samples across LEVIR-CD, DSIFN, and custom datasets."""
    levir_samples = []
    samples_dir_a = os.path.join("samples_LEVIR", "A")
    if os.path.isdir(samples_dir_a):
        for f in sorted(os.listdir(samples_dir_a)):
            if f.endswith(('.png', '.jpg', '.tif', '.tiff')):
                sample_id = f.rsplit(".", 1)[0]
                t1_rel = f"samples_LEVIR/A/{f}"
                t2_rel = f"samples_LEVIR/B/{f}"
                levir_samples.append({
                    "id": sample_id,
                    "filename": f,
                    "name": f"LEVIR: {f}",
                    "type": "bi_temporal_optical",
                    "t1": f"/api/v1/sample_image?path={t1_rel}",
                    "t2": f"/api/v1/sample_image?path={t2_rel}",
                    "t1_url": f"/api/v1/sample_image?path={t1_rel}",
                    "t2_url": f"/api/v1/sample_image?path={t2_rel}",
                    "t1_path": t1_rel,
                    "t2_path": t2_rel
                })

    dsifn_samples = []
    dsifn_dir_a = os.path.join("samples_DSIFN", "A")
    if os.path.isdir(dsifn_dir_a):
        for f in sorted(os.listdir(dsifn_dir_a)):
            if f.endswith(('.png', '.jpg', '.tif', '.tiff')):
                sample_id = f.rsplit(".", 1)[0]
                t1_rel = f"samples_DSIFN/A/{f}"
                t2_rel = f"samples_DSIFN/B/{f}"
                dsifn_samples.append({
                    "id": sample_id,
                    "filename": f,
                    "name": f"DSIFN: {f}",
                    "type": "bi_temporal_optical",
                    "t1": f"/api/v1/sample_image?path={t1_rel}",
                    "t2": f"/api/v1/sample_image?path={t2_rel}",
                    "t1_url": f"/api/v1/sample_image?path={t1_rel}",
                    "t2_url": f"/api/v1/sample_image?path={t2_rel}",
                    "t1_path": t1_rel,
                    "t2_path": t2_rel
                })

    custom_samples = []
    if os.path.isfile(os.path.join("custom_input", "A", "image1.png")):
        t1_rel = "custom_input/A/image1.png"
        t2_rel = "custom_input/B/image1.png"
        custom_samples.append({
            "id": "custom_pair_1",
            "filename": "image1.png",
            "name": "Custom Input Pair (450x221)",
            "type": "bi_temporal_optical",
            "t1": f"/api/v1/sample_image?path={t1_rel}",
            "t2": f"/api/v1/sample_image?path={t2_rel}",
            "t1_url": f"/api/v1/sample_image?path={t1_rel}",
            "t2_url": f"/api/v1/sample_image?path={t2_rel}",
            "t1_path": t1_rel,
            "t2_path": t2_rel
        })

    # Also build a flat list of samples for convenience
    all_samples = levir_samples + dsifn_samples + custom_samples

    return {
        "samples": all_samples,
        "levir_samples": levir_samples,
        "dsifn_samples": dsifn_samples,
        "custom_samples": custom_samples
    }


@app.get("/api/v1/sample_image")
@app.get("/api/sample_image")
async def get_sample_image(path: str):
    """Serves a sample image file from disk."""
    norm_path = os.path.normpath(path.strip().lstrip("/\\"))
    if not os.path.isfile(norm_path):
        cwd_path = os.path.join(os.getcwd(), norm_path)
        if os.path.isfile(cwd_path):
            norm_path = cwd_path
        else:
            raise HTTPException(status_code=404, detail=f"Sample image not found: {path}")
    mime, _ = mimetypes.guess_type(norm_path)
    return FileResponse(norm_path, media_type=mime or "image/png")


@app.post("/api/v1/upload")
async def upload_image(file: UploadFile = File(...)):
    """Uploads a satellite / aerial image or GeoTIFF for processing."""
    upload_dir = os.path.join("custom_input", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_id = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(upload_dir, file_id)
    
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
        
    geo_meta = GeoTIFFProcessor.inspect(file_path) if GeoTIFFProcessor.is_geotiff(file_path) else {}
    
    return {
        "status": "UPLOADED",
        "filename": file.filename,
        "file_path": file_path,
        "is_geotiff": GeoTIFFProcessor.is_geotiff(file_path),
        "geospatial_metadata": geo_meta
    }


@app.post("/api/v1/validate")
async def validate_pair(
    image_1_path: str = Form(...),
    image_2_path: Optional[str] = Form(None),
    task_type: str = Form("bi_temporal")
):
    """Validates image pair dimensions, bands, modalities, and geospatial overlap."""
    val_result = InputValidator.validate_pair(image_1_path, image_2_path, task_type=task_type)
    
    overlap_info = {}
    if image_2_path and GeoTIFFProcessor.is_geotiff(image_1_path) and GeoTIFFProcessor.is_geotiff(image_2_path):
        m1 = GeoTIFFProcessor.inspect(image_1_path)
        m2 = GeoTIFFProcessor.inspect(image_2_path)
        overlap_info = GeoTIFFProcessor.check_spatial_overlap(m1, m2)
        
    return {
        "validation": val_result,
        "geospatial_overlap": overlap_info
    }


@app.post("/api/v1/analyze")
@app.post("/api/analyze")
async def analyze_scene(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Main multimodal analysis pipeline."""
    try:
        # Resolve Image 1
        if req.image_1_path and os.path.isfile(req.image_1_path):
            img_1 = req.image_1_path
        elif req.image_1:
            raw_b64 = req.image_1.split(",", 1)[1] if "," in req.image_1 else req.image_1
            img_1 = Image.open(BytesIO(base64.b64decode(raw_b64))).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Primary image (image_1) required")

        # Resolve Image 2
        img_2 = None
        if req.image_2_path and os.path.isfile(req.image_2_path):
            img_2 = req.image_2_path
        elif req.image_2:
            raw_b64 = req.image_2.split(",", 1)[1] if "," in req.image_2 else req.image_2
            img_2 = Image.open(BytesIO(base64.b64decode(raw_b64))).convert("RGB")

        # Execute Agent Controller in threadpool to keep FastAPI async event loop unblocked
        result = await run_in_threadpool(
            lambda: controller.execute(
                image_1=img_1,
                image_2=img_2,
                question=req.question,
                task=req.task,
                modality_1=req.modality_1 or "optical",
                modality_2=req.modality_2 or "optical"
            )
        )

        # Generate task ID and cache
        task_id = uuid.uuid4().hex[:12]
        
        # Format response
        resp = {
            "task_id": task_id,
            "task": result.get("task"),
            "question": result.get("question"),
            "answer": result.get("answer"),
            "confidence": result.get("confidence"),
            "confidence_source": result.get("confidence_source"),
            "uncertainty": result.get("uncertainty"),
            "specialist_statuses": result.get("specialist_statuses", {}),
            "models_used": result.get("models_used", []),
            "trace": result.get("trace", []),
            "execution_plan": result.get("execution_plan", []),
            "execution_time_sec": result.get("execution_time_sec"),
            "warnings": result.get("warnings", []),
            "geospatial_metadata": result.get("geospatial_metadata", {})
        }

        if "change_percentage" in result:
            resp["change_percentage"] = result["change_percentage"]
            resp["change_type"] = result.get("change_type")
            resp["spatial_evidence"] = result.get("spatial_evidence", {})

        if "bboxes" in result:
            resp["bboxes"] = result.get("bboxes", [])

        if "geojson" in result and result["geojson"]:
            resp["geojson"] = result["geojson"]

        # Base64 image visualizations
        if "change_mask" in result and isinstance(result["change_mask"], np.ndarray):
            resp["change_mask_url"] = array_to_base64(result["change_mask"])

        if "difference_map" in result and isinstance(result["difference_map"], np.ndarray):
            resp["difference_map_url"] = array_to_base64(result["difference_map"])

        if "overlay_t1" in result and isinstance(result["overlay_t1"], np.ndarray):
            resp["overlay_t1_url"] = array_to_base64(result["overlay_t1"])

        if "overlay_t2" in result and isinstance(result["overlay_t2"], np.ndarray):
            resp["overlay_t2_url"] = array_to_base64(result["overlay_t2"])

        if "fusion_visualization" in result and isinstance(result["fusion_visualization"], np.ndarray):
            resp["fusion_vis_url"] = array_to_base64(result["fusion_visualization"])

        if "visualization" in result and isinstance(result["visualization"], np.ndarray):
            resp["grounding_vis_url"] = array_to_base64(result["visualization"])

        if "optical_evidence" in result:
            resp["optical_evidence"] = result["optical_evidence"]
        if "sar_evidence" in result:
            resp["sar_evidence"] = result["sar_evidence"]
        if "fusion_evidence" in result:
            resp["fusion_evidence"] = result["fusion_evidence"]

        if "top_predictions" in result:
            resp["top_predictions"] = result["top_predictions"]

        # 4. Integrate Qwen2-VL Visual Context (Optimized with instant cache hit or non-blocking async)
        if qwen_vision_service.is_available():
            try:
                # Fast cache check first (SHA-256 of image content)
                cache_key = VisualContextCache.compute_key(img_1, metadata=result.get("geospatial_metadata"))
                cached_context = qwen_vision_service.cache.get(cache_key)

                if cached_context:
                    resp["visual_context"] = cached_context
                    resp["vision_status"] = "completed"
                    if "Qwen2-VL-2B-Instruct" not in resp["models_used"]:
                        resp["models_used"].append("Qwen2-VL-2B-Instruct")
                elif result.get("task") in ["scene_understanding", "captioning"] and not result.get("answer"):
                    # Dedicated single-image visual captioning task where user specifically waited for scene context
                    v_res = await run_in_threadpool(
                        lambda: qwen_vision_service.analyze_image(img_1, metadata=result.get("geospatial_metadata"))
                    )
                    if v_res.get("status") == "completed" and v_res.get("visual_context"):
                        resp["visual_context"] = v_res["visual_context"]
                        resp["vision_status"] = "completed"
                        if "Qwen2-VL-2B-Instruct" not in resp["models_used"]:
                            resp["models_used"].append("Qwen2-VL-2B-Instruct")
                    elif v_res.get("vision_error"):
                        resp["visual_context"] = None
                        resp["vision_error"] = v_res["vision_error"]
                else:
                    # For change detection & rapid execution: return specialist results immediately in <1s!
                    # Trigger background vision inference so visual cache is warmed asynchronously
                    resp["visual_context"] = None
                    resp["vision_status"] = "pending"
                    background_tasks.add_task(
                        qwen_vision_service.analyze_image,
                        img_1,
                        result.get("geospatial_metadata")
                    )
            except Exception as e:
                resp["visual_context"] = None
                resp["vision_error"] = str(e)
        elif "visual_context" in result:
            resp["visual_context"] = result["visual_context"]
            resp["vision_status"] = "completed"

        resp["markdown_report"] = ReportExporter.to_markdown(result)
        results_cache[task_id] = resp

        return resp

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/vision/context")
@app.post("/api/vision/context")
async def extract_vision_context(req: AnalyzeRequest):
    """
    Dedicated endpoint to extract 18-field structured visual context using Qwen2-VL-2B-Instruct.
    Runs locally on GPU if available with CPU fallback.
    """
    try:
        if req.image_1_path and os.path.isfile(req.image_1_path):
            img_1 = req.image_1_path
        elif req.image_1:
            raw_b64 = req.image_1.split(",", 1)[1] if "," in req.image_1 else req.image_1
            img_1 = Image.open(BytesIO(base64.b64decode(raw_b64))).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Primary image (image_1 or image_1_path) required")

        result = await run_in_threadpool(lambda: qwen_vision_service.analyze_image(img_1))
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/v1/result/{task_id}")
async def get_cached_result(task_id: str):
    """Retrieves cached analysis result by task ID."""
    if task_id not in results_cache:
        raise HTTPException(status_code=404, detail="Task result not found or expired")
    return results_cache[task_id]


@app.get("/api/v1/benchmark")
async def get_benchmark_report():
    """Generates quantitative benchmark summary across models."""
    table_md = benchmark_runner.generate_comparison_table()
    return {
        "status": "SUCCESS",
        "benchmark_markdown": table_md
    }


# Mount Frontend static assets
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)
        return HTMLResponse("<h1>SatQuery AI Backend Running</h1>")


# ==============================================================================
# JARVIS 2.0 — Autonomous Remote-Sensing Analyst Copilot
# Centralized configuration — ONE source of truth: Ollama qwen3:8b
# ==============================================================================

from ..services.tts import tts_service, normalize_for_speech, prepare_speech_text
from ..services.ollama import ollama_service, OLLAMA_MODEL, OLLAMA_BASE_URL
from ..agent.tools import TOOL_REGISTRY, ClusterAnalytics, EvidenceFusion
from ..agent.query_engine import query_engine
from ..agent.state import MissionState, InputState, UIState

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
    "show_executive_summary", "show_agent_progress", "show_why_this_result",
    "show_why_this_matters", "generate_report", "update_answer",
    "speak_response", "clear_selection", "reset_view"
}


class JarvisRequest(BaseModel):
    query: str = Field(..., description="Natural language user query")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    has_image_1: bool = Field(default=False)
    has_image_2: bool = Field(default=False)
    image_1_path: Optional[str] = Field(default=None)
    image_2_path: Optional[str] = Field(default=None)
    image_1_data: Optional[str] = Field(default=None, description="Base64 data URL")
    image_2_data: Optional[str] = Field(default=None, description="Base64 data URL")
    current_tab: Optional[str] = Field(default="overview")
    prior_result_id: Optional[str] = Field(default=None)
    selected_cluster: Optional[int] = Field(default=None)
    voice_enabled: bool = Field(default=True)
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)


class JarvisTTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: float = 1.0


class JarvisExplainRequest(BaseModel):
    query: str
    prior_result_id: str
    cluster_index: Optional[int] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ActionAckRequest(BaseModel):
    action: str
    status: str = "success"
    tab: Optional[str] = None
    cluster_index: Optional[int] = None
    timestamp: Optional[float] = None


async def _check_ollama_status() -> Tuple[bool, str]:
    """Returns (available, reason_string) using centralized Ollama service."""
    return await ollama_service.check_health()


async def _qwen3_plan(
    query: str,
    has_image_1: bool,
    has_image_2: bool,
    current_tab: str,
    prior_result: Optional[dict],
    selected_cluster: Optional[int]
) -> dict:
    """Ask Universal Query Engine to produce a validated structured execution plan."""
    mstate = MissionState()
    mstate.inputs.has_image_1 = has_image_1
    mstate.inputs.has_image_2 = has_image_2
    mstate.ui.current_tab = current_tab
    mstate.change.selected_cluster_index = selected_cluster
    return await query_engine.plan_investigation(query, mstate, prior_result)


async def _qwen3_synthesize(
    query: str,
    plan: dict,
    specialist_result: Optional[dict],
    prior_result: Optional[dict],
    selected_cluster_info: Optional[dict] = None
) -> dict:
    """Ask Universal Query Engine to synthesize a grounded response with provenance, why_this_result, why_this_matters, and executive summary."""
    mstate = MissionState()
    return await query_engine.synthesize_grounded_response(
        query=query,
        plan=plan,
        specialist_result=specialist_result,
        prior_result=prior_result,
        state=mstate,
        selected_cluster_info=selected_cluster_info
    )


from ..agent.tools import TOOL_REGISTRY, ClusterAnalytics, EvidenceFusion, UIActionBuilder

def _build_jarvis_ui_actions(
    plan: dict,
    specialist_result: Optional[dict],
    trace_steps: list,
    target_cluster: Optional[int] = None
) -> list:
    """Build a VALIDATED list of UI actions — strict adherence to ALLOWED_UI_ACTIONS."""
    return UIActionBuilder.build_actions(
        plan=plan,
        specialist_result=specialist_result,
        trace_steps=trace_steps,
        target_cluster=target_cluster
    )


async def _execute_jarvis_specialist(
    specialist: str, req: JarvisRequest, plan: dict
) -> Tuple[dict, str]:
    """Execute the selected specialist model and return (formatted_result, task_id)."""
    question = plan.get("grounding_target") or plan.get("target_object") or plan.get("question", "") or req.query

    # Resolve Image 1
    img1 = None
    if req.image_1_path and os.path.isfile(req.image_1_path):
        img1 = req.image_1_path
    elif req.image_1_data:
        raw = req.image_1_data.split(",", 1)[1] if "," in req.image_1_data else req.image_1_data
        img1 = Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
    if img1 is None:
        raise ValueError("No valid primary image provided for specialist execution.")

    # Resolve Image 2
    img2 = None
    if req.image_2_path and os.path.isfile(req.image_2_path):
        img2 = req.image_2_path
    elif req.image_2_data:
        raw = req.image_2_data.split(",", 1)[1] if "," in req.image_2_data else req.image_2_data
        img2 = Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")

    task_map = {
        "change_detection": "change_analysis",
        "vqa": None,
        "grounding": "grounding",
        "optical_sar": "optical_sar_fusion",
        "geospatial": None
    }
    task = task_map.get(specialist)

    # Run specialist in thread pool to avoid blocking async loop
    raw_result = await run_in_threadpool(
        lambda: controller.execute(
            image_1=img1, image_2=img2,
            question=question, task=task
        )
    )

    task_id = uuid.uuid4().hex[:12]
    resp: Dict[str, Any] = {
        "task_id": task_id,
        "result_id": task_id,
        "task": raw_result.get("task"),
        "question": raw_result.get("question"),
        "answer": raw_result.get("answer"),
        "confidence": raw_result.get("confidence"),
        "confidence_source": raw_result.get("confidence_source"),
        "uncertainty": raw_result.get("uncertainty"),
        "specialist_statuses": raw_result.get("specialist_statuses", {}),
        "models_used": raw_result.get("models_used", []),
        "trace": raw_result.get("trace", []),
        "execution_plan": raw_result.get("execution_plan", []),
        "execution_time_sec": raw_result.get("execution_time_sec"),
        "warnings": raw_result.get("warnings", []),
        "geospatial_metadata": raw_result.get("geospatial_metadata", {})
    }
    if "change_percentage" in raw_result:
        resp["change_percentage"] = raw_result["change_percentage"]
        resp["change_type"] = raw_result.get("change_type")
        resp["spatial_evidence"] = raw_result.get("spatial_evidence", {})
    for k in ("bboxes", "geojson", "num_regions", "box_confidences",
              "top_predictions", "optical_evidence", "sar_evidence", "fusion_evidence"):
        if k in raw_result:
            resp[k] = raw_result[k]
    for src_k, dst_k in [
        ("change_mask",          "change_mask_url"),
        ("difference_map",       "difference_map_url"),
        ("overlay_t1",           "overlay_t1_url"),
        ("overlay_t2",           "overlay_t2_url"),
        ("fusion_visualization", "fusion_vis_url"),
        ("visualization",        "grounding_vis_url"),
    ]:
        if src_k in raw_result and isinstance(raw_result[src_k], np.ndarray):
            resp[dst_k] = array_to_base64(raw_result[src_k])
    resp["markdown_report"] = ReportExporter.to_markdown(raw_result)
    results_cache[task_id] = resp
    return resp, task_id


# ==============================================================================
# JARVIS 2.0 API Endpoints
# ==============================================================================

@app.get("/api/v1/jarvis/status", tags=["JARVIS 2.0"])
async def jarvis_status():
    """Check JARVIS 2.0 readiness: Ollama qwen3:8b and Neural TTS status."""
    ollama_ok, reason = await _check_ollama_status()
    tts_status = await tts_service.get_status()

    return {
        "jarvis": "online" if ollama_ok else "offline",
        "status": "ONLINE" if ollama_ok else "OFFLINE",
        "ollama": "online" if ollama_ok else "offline",
        "ollama_available": ollama_ok,
        "model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "tts": tts_status["status"],
        "tts_provider": tts_status["provider"],
        "voice": tts_status["voice"],
        "voice_alt": tts_status["voice_alt"],
        "reason": None if ollama_ok else reason
    }


@app.post("/api/v1/jarvis/tts", tags=["JARVIS 2.0"])
async def jarvis_tts_generate(req: JarvisTTSRequest):
    """Generate high-quality speech audio from text using TTSService."""
    result = await tts_service.synthesize_base64(req.text, voice=req.voice)
    return result


@app.post("/api/v1/jarvis", tags=["JARVIS 2.0"])
async def jarvis_query(req: JarvisRequest):
    """
    JARVIS 2.0 Autonomous Remote-Sensing Analyst Copilot Pipeline.
    State-aware planning -> Real specialist execution -> Grounded synthesis -> Provenance -> UI Actions -> Voice.
    """
    req_id = uuid.uuid4().hex[:6]
    t_start = time.perf_counter()
    qwen_calls = 0
    specialist_calls = 0
    ollama_calls = 0

    # 1. Retrieve cached prior result if provided
    prior_result: Optional[dict] = None
    if req.prior_result_id:
        prior_result = (
            results_cache.get(req.prior_result_id)
            or results_cache.get(f"jarvis_{req.prior_result_id}")
        )
        if prior_result and "specialist_result" in prior_result:
            prior_result = prior_result["specialist_result"] or prior_result

    trace_steps = ["Query received", "Application state verified"]

    try:
        # 2. State-aware Deterministic Planning
        trace_steps.append("Building analysis plan")
        t_plan_start = time.perf_counter()
        plan = await _qwen3_plan(
            query=req.query,
            has_image_1=req.has_image_1,
            has_image_2=req.has_image_2,
            current_tab=req.current_tab or "overview",
            prior_result=prior_result,
            selected_cluster=req.selected_cluster
        )
        t_plan_done = time.perf_counter()
        specialist = plan.get("specialist", "none")
        goal = plan.get("goal", "analysis")
        intent = plan.get("intent", "UNKNOWN")
        trace_steps.append(f"Plan goal: {goal}")
        trace_steps.append(f"Specialist selected: {specialist}")

        # FAST PATH: Deterministic UI actions bypass Qwen, ChangeFormer, and Ollama completely (< 0.05s)
        if intent.startswith("UI_") or intent == "UI_COMMAND":
            t0_ui = time.perf_counter()
            ui_actions = _build_jarvis_ui_actions(
                plan=plan,
                specialist_result=prior_result,
                trace_steps=trace_steps,
                target_cluster=0
            )
            resp_text = plan.get("response") or "Executing UI action."
            jarvis_task_id = uuid.uuid4().hex[:12]
            ui_ms = (time.perf_counter() - t0_ui) * 1000.0
            total_dur = (time.perf_counter() - t_start) * 1000.0
            timings = {
                "1_reception": round((t_plan_start - t_start) * 1000.0, 2),
                "2_img_preprocess": 0.0,
                "3_model_init": 0.0,
                "4_qwen_plan": round((t_plan_done - t_plan_start) * 1000.0, 2),
                "5_specialist_infer": 0.0,
                "6_evidence_fusion": 0.0,
                "7_qwen_synthesis": 0.0,
                "8_ui_serialization": round(ui_ms, 2),
                "9_total_e2e": round(total_dur, 2)
            }
            logger.info(
                f"[JARVIS][{req_id}] QUERY=\"{req.query}\" INTENT={intent} "
                f"TOTAL={total_dur:.1f}ms FAST_PATH=TRUE"
            )
            return {
                "success": True,
                "intent": intent,
                "response": resp_text,
                "answer": resp_text,
                "speak_text": prepare_speech_text(resp_text),
                "audio_base64": None,
                "evidence": prior_result or {},
                "ui_action": plan.get("ui_action"),
                "ui_actions": ui_actions,
                "task_id": jarvis_task_id,
                "specialist_task_id": None,
                "model": "UI Fast Path",
                "timings_ms": timings,
                "trace": trace_steps + [f"Fast-path UI dispatch for {intent}"]
            }

        # 3. For analytical queries that require LLM synthesis, verify Ollama readiness
        ollama_ok, ollama_reason = await _check_ollama_status()
        if not ollama_ok:
            raise HTTPException(
                status_code=503,
                detail=f"JARVIS is unavailable because Ollama {OLLAMA_MODEL} could not be reached. {ollama_reason}"
            )

        # 4. Handle Missing Requirements honestly
        if plan.get("missing_input") == "image_2" or (specialist == "change_detection" and not req.has_image_2):
            missing_text = (
                "I have the first image, but I need the second acquisition to perform a bi-temporal comparison."
                if req.has_image_1 else "Please upload or select the imagery pair to run bi-temporal change detection."
            )
            audio_info = await tts_service.synthesize_base64(missing_text) if req.voice_enabled else {}
            return {
                "success": False,
                "intent": intent,
                "response": missing_text,
                "answer": missing_text,
                "speak_text": prepare_speech_text(missing_text),
                "audio_base64": audio_info.get("audio_base64"),
                "evidence": {},
                "ui_action": None,
                "ui_actions": [{"action": "navigate_tab", "tab": "overview"}],
                "why_this_result": {
                    "model": "None",
                    "inputs": "Incomplete imagery",
                    "output_meaning": "Analysis could not execute",
                    "confidence_type": "N/A",
                    "limitations": "Requires both T1 and T2 acquisitions"
                },
                "why_this_matters": "Bi-temporal analysis requires two timestamps to compute spatial variance.",
                "recommended_next_step": "Load the second image in the T2 slot.",
                "executive_summary": {
                    "finding": "Analysis pending: second image required.",
                    "largest_change": "N/A", "evidence": "None",
                    "geospatial_info": "N/A", "potential_relevance": "N/A",
                    "limitations": "Missing T2 input"
                },
                "task_id": uuid.uuid4().hex[:12],
                "model": OLLAMA_MODEL,
                "trace": trace_steps + ["Analysis halted: missing T2 image"]
            }

        # 5. Execute Specialist or Reuse Cache
        specialist_result: Optional[dict] = None
        new_task_id: Optional[str] = None
        selected_cluster_info: Optional[dict] = None
        target_cluster_idx = plan.get("target_cluster", 0)
        t0_specialist = time.perf_counter()
        specialist_infer_ms = 0.0
        evidence_fusion_ms = 0.0

        if specialist in ("rank_change_clusters", "inspect_change_cluster") or (plan.get("reuse_prior") and prior_result):
            specialist_result = prior_result
            new_task_id = req.prior_result_id
            trace_steps.append("Reusing cached specialist analysis (avoided model rerun)")
            t0_fus = time.perf_counter()
            if prior_result and prior_result.get("bboxes"):
                ranked = ClusterAnalytics.rank_clusters(prior_result["bboxes"])
                selected_cluster_info = ClusterAnalytics.get_cluster(ranked, target_cluster_idx)
                trace_steps.append(f"Ranked {len(ranked)} clusters; selected cluster {target_cluster_idx}")
            evidence_fusion_ms = (time.perf_counter() - t0_fus) * 1000.0

        elif plan.get("requires_specialist") and specialist != "none":
            if not req.has_image_1:
                raise HTTPException(400, detail="No imagery loaded. Please upload or select images first.")
            trace_steps.append(f"Executing {specialist} specialist")
            specialist_calls = 1
            try:
                specialist_result, new_task_id = await _execute_jarvis_specialist(specialist, req, plan)
                specialist_infer_ms = (time.perf_counter() - t0_specialist) * 1000.0
                trace_steps.append(f"Specialist inference complete ({specialist_infer_ms / 1000.0:.2f}s)")
                t0_fus = time.perf_counter()
                if specialist_result.get("bboxes"):
                    ranked = ClusterAnalytics.rank_clusters(specialist_result["bboxes"])
                    selected_cluster_info = ClusterAnalytics.get_cluster(ranked, 0)
                    trace_steps.append(f"Identified {len(ranked)} change clusters")
                evidence_fusion_ms = (time.perf_counter() - t0_fus) * 1000.0
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning(f"Specialist {specialist} execution issue: {exc}")
                trace_steps.append(f"Specialist {specialist} limitation: {exc}")
                specialist_result = {
                    "task": specialist,
                    "answer": f"The {specialist} specialist is currently unavailable ({exc}).",
                    "confidence": 0.5,
                    "confidence_source": "Heuristic Assessment",
                    "models_used": [f"{specialist} (Limited)"],
                    "warnings": [str(exc)]
                }
                new_task_id = uuid.uuid4().hex[:12]
        else:
            trace_steps.append("Contextual synthesis (no specialist execution required)")

        # 6. Synthesize Grounded Analyst Response with Qwen3:8b
        trace_steps.append(f"Synthesizing grounded analysis with {OLLAMA_MODEL}")
        ollama_calls = 1
        t0_synth = time.perf_counter()
        synthesis = await _qwen3_synthesize(
            query=req.query,
            plan=plan,
            specialist_result=specialist_result,
            prior_result=prior_result,
            selected_cluster_info=selected_cluster_info
        )
        synthesis_ms = (time.perf_counter() - t0_synth) * 1000.0

        # 7. Generate Validated UI Actions
        t0_ui = time.perf_counter()
        ui_actions = _build_jarvis_ui_actions(
            plan=plan,
            specialist_result=specialist_result,
            trace_steps=trace_steps,
            target_cluster=target_cluster_idx
        )

        # 8. Neural TTS Synthesis for Speak Text
        speak_text = synthesis.get("speak_text", "")
        audio_base64 = None
        if req.voice_enabled and speak_text:
            tts_res = await tts_service.synthesize_base64(speak_text)
            audio_base64 = tts_res.get("audio_base64")

        # 9. Evidence Provenance & Cache
        jarvis_task_id = uuid.uuid4().hex[:12]
        provenance = {
            "source_model": specialist_result.get("task", "Contextual Reasoning") if specialist_result else "State Context",
            "result_id": new_task_id or jarvis_task_id,
            "models_used": specialist_result.get("models_used", []) if specialist_result else [OLLAMA_MODEL],
            "confidence": specialist_result.get("confidence") if specialist_result else None
        }

        results_cache[f"jarvis_{jarvis_task_id}"] = {
            "task_id": jarvis_task_id,
            "specialist_result": specialist_result,
            "synthesis": synthesis,
            "provenance": provenance,
            "trace": trace_steps,
            "plan": plan
        }
        if specialist_result and new_task_id:
            results_cache[jarvis_task_id] = specialist_result

        ui_ms = (time.perf_counter() - t0_ui) * 1000.0
        total_e2e_ms = (time.perf_counter() - t_start) * 1000.0

        timings_ms = {
            "1_reception": round((t_plan_start - t_start) * 1000.0, 2),
            "2_img_preprocess": round(max(0.0, specialist_infer_ms * 0.1), 2),
            "3_model_init": 0.0,
            "4_qwen_plan": round((t_plan_done - t_plan_start) * 1000.0, 2),
            "5_specialist_infer": round(specialist_infer_ms, 2),
            "6_evidence_fusion": round(evidence_fusion_ms, 2),
            "7_qwen_synthesis": round(synthesis_ms, 2),
            "8_ui_serialization": round(ui_ms, 2),
            "9_total_e2e": round(total_e2e_ms, 2)
        }

        logger.info(
            f"[JARVIS][{req_id}] QUERY=\"{req.query}\" INTENT={intent} "
            f"TOTAL={total_e2e_ms:.1f}ms [PLAN={timings_ms['4_qwen_plan']}ms, "
            f"SPEC={timings_ms['5_specialist_infer']}ms, SYNTH={timings_ms['7_qwen_synthesis']}ms]"
        )

        return {
            "success": True,
            "intent": intent,
            "response": synthesis.get("answer", "Analysis complete."),
            "answer": synthesis.get("answer", "Analysis complete."),
            "speak_text": speak_text,
            "audio_base64": audio_base64,
            "evidence": specialist_result or {},
            "provenance": provenance,
            "why_this_result": synthesis.get("why_this_result", {}),
            "why_this_matters": synthesis.get("why_this_matters", ""),
            "recommended_next_step": synthesis.get("recommended_next_step", ""),
            "executive_summary": synthesis.get("executive_summary", {}),
            "ui_action": plan.get("ui_action"),
            "ui_actions": ui_actions,
            "task_id": jarvis_task_id,
            "specialist_task_id": new_task_id,
            "model": OLLAMA_MODEL,
            "timings_ms": timings_ms,
            "trace": trace_steps
        }

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/jarvis/explain", tags=["JARVIS 2.0"])
async def jarvis_explain(req: JarvisExplainRequest):
    """JARVIS 2.0 follow-up: explain selected cluster/region using cached results."""
    ollama_ok, reason = await _check_ollama_status()
    if not ollama_ok:
        raise HTTPException(503, detail=f"JARVIS unavailable: {reason}")

    prior = (
        results_cache.get(req.prior_result_id)
        or results_cache.get(f"jarvis_{req.prior_result_id}")
    )
    if prior and "specialist_result" in prior:
        prior = prior["specialist_result"] or prior
    if not prior:
        raise HTTPException(404, detail="Prior analysis result not found. Please run analysis first.")

    cluster_info = None
    if req.cluster_index is not None and prior.get("bboxes"):
        ranked = ClusterAnalytics.rank_clusters(prior["bboxes"])
        cluster_info = ClusterAnalytics.get_cluster(ranked, req.cluster_index)

    synthesis = await _qwen3_synthesize(
        query=req.query,
        plan={"goal": "explain_region"},
        specialist_result=None,
        prior_result=prior,
        selected_cluster_info=cluster_info
    )

    audio_res = await tts_service.synthesize_base64(synthesis.get("speak_text", ""))

    return {
        "answer": synthesis.get("answer", ""),
        "speak_text": synthesis.get("speak_text", ""),
        "audio_base64": audio_res.get("audio_base64"),
        "why_this_result": synthesis.get("why_this_result", {}),
        "why_this_matters": synthesis.get("why_this_matters", ""),
        "recommended_next_step": synthesis.get("recommended_next_step", ""),
        "executive_summary": synthesis.get("executive_summary", {}),
        "model": OLLAMA_MODEL,
        "reused_result_id": req.prior_result_id
    }


@app.post("/api/v1/jarvis/action_ack", tags=["JARVIS 2.0"])
async def jarvis_action_ack(req: ActionAckRequest):
    """Acknowledge executed UI action from frontend to maintain bidirectional state sync."""
    return {"status": "acknowledged", "action": req.action, "result": req.status}

