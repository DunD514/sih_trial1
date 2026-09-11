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
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import httpx
from fastapi.concurrency import run_in_threadpool

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


# Initialize FastAPI app
app = FastAPI(
    title="SatQuery AI API",
    description="Interactive Multimodal Remote-Sensing AI Assistant API (SIH26167)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
@app.get("/api/v1/health")
@app.get("/api/health")
async def health_check():
    """Health check and model readiness status."""
    def _status(model, error_attr):
        if model is not None:
            return "Loaded"
        err = getattr(controller, error_attr, None)
        return f"Unavailable — {err}" if err else "Not yet loaded"

    return {
        "status": "healthy",
        "system": "SatQuery AI Remote Sensing Assistant",
        "version": "2.0.0",
        "models": {
            "change_detection": _status(controller._change_model, "_change_model_error"),
            "vqa": _status(controller._vqa, "_vqa_error"),
            "grounding": _status(controller._grounding, "_grounding_error"),
            "optical_sar": _status(controller._optical_sar, "_optical_sar_error"),
            "geospatial": "Rasterio CRS Engine (Ready)"
        }
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
async def analyze_scene(req: AnalyzeRequest):
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

        # Execute Agent Controller
        result = controller.execute(
            image_1=img_1,
            image_2=img_2,
            question=req.question,
            task=req.task,
            modality_1=req.modality_1 or "optical",
            modality_2=req.modality_2 or "optical"
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

        resp["markdown_report"] = ReportExporter.to_markdown(result)
        results_cache[task_id] = resp

        return resp

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
from ..agent.tools import TOOL_REGISTRY, ClusterAnalytics

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen3:8b")  # STRICT: ONLY qwen3:8b

ALLOWED_SPECIALIST_TOOLS = {
    "change_detection", "vqa", "grounding", "optical_sar",
    "geospatial", "rank_change_clusters", "inspect_change_cluster",
    "build_summary", "build_report", "none"
}

ALLOWED_UI_ACTIONS = {
    "navigate_tab", "show_overlay", "show_side_by_side", "show_change_mask",
    "show_diff_map", "select_cluster", "highlight_cluster", "zoom_to_cluster",
    "show_grounding", "show_vqa", "show_optical_sar", "show_geospatial",
    "open_trace", "open_evidence", "open_impact", "open_summary",
    "generate_report", "show_why_this_result", "show_why_this_matters",
    "show_executive_summary", "show_agent_progress", "update_answer",
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


def _extract_json_from_llm(text: str) -> dict:
    """Strip Qwen3 <think> blocks and extract the JSON object."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for pattern in [r"\{[\s\S]*\}", r"\{.*?\}"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return _json.loads(m.group())
            except _json.JSONDecodeError:
                continue
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        raise ValueError(f"No valid JSON in LLM response: {text[:300]}")


async def _call_qwen3(messages: list, timeout: float = 120.0) -> str:
    """Call Ollama qwen3:8b API and return the assistant text."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "think": False,  # Disable thinking mode — all tokens go to response
                "options": {"temperature": 0.1, "num_predict": 512}
            }
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def _check_ollama_status() -> Tuple[bool, str]:
    """Returns (available, reason_string)."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                return True, "OK"
            short_list = ", ".join(models[:6]) or "none"
            return False, f"{OLLAMA_MODEL} not found in Ollama. Available: {short_list}"
    except httpx.ConnectError:
        return False, f"Ollama server not reachable at {OLLAMA_BASE_URL}"
    except Exception as exc:
        return False, str(exc)


async def _qwen3_plan(
    query: str,
    has_image_1: bool,
    has_image_2: bool,
    current_tab: str,
    prior_result: Optional[dict],
    selected_cluster: Optional[int]
) -> dict:
    """Ask Qwen3 to produce a validated structured execution plan based on application state."""
    prior_summary = "None"
    ranked_clusters = []
    if prior_result:
        task = prior_result.get("task", "unknown")
        bboxes = prior_result.get("bboxes", [])
        ranked_clusters = ClusterAnalytics.rank_clusters(bboxes) if bboxes else []
        if task == "change_analysis":
            pct = prior_result.get("change_percentage", "?")
            n = len(bboxes)
            prior_summary = f"change_detection: {pct}% change, {n} cluster(s). Largest cluster index: {ranked_clusters[0]['index'] if ranked_clusters else 0}"
        elif task in ("scene_vqa", "numerical_vqa", "presence_vqa"):
            prior_summary = f"vqa: answer={prior_result.get('answer', '?')}"
        elif task == "grounding":
            prior_summary = f"grounding: {prior_result.get('num_regions', 0)} region(s)"
        elif task == "optical_sar_fusion":
            prior_summary = f"optical_sar: {str(prior_result.get('answer', ''))[:80]}"
        else:
            prior_summary = f"{task}: completed"

    system_prompt = (
        "You are the autonomous AI Analyst orchestrator for SatQuery AI, an aerospace remote sensing workstation.\n"
        "Your role: analyze user requests against current application state, select controlled tools, and produce an execution plan.\n\n"
        "AVAILABLE TOOLS (STRICT REGISTRY — USE EXACT IDENTIFIERS ONLY):\n"
        '- "change_detection": ChangeFormerV6 bi-temporal transformer. Requires has_image_2=true.\n'
        '- "vqa": ViLT-B32 visual question answering. Works with 1 image.\n'
        '- "grounding": OWL-ViT zero-shot spatial grounding bounding boxes. Works with 1 image.\n'
        '- "optical_sar": Rule-based Optical + SAR dual-sensor fusion. Needs 2 images.\n'
        '- "geospatial": Rasterio GeoTIFF CRS and coordinate extraction.\n'
        '- "rank_change_clusters": Ranks cached ChangeFormer clusters by pixel area.\n'
        '- "inspect_change_cluster": Extracts details for a specific cluster.\n'
        '- "build_summary": Compiles executive summary from existing evidence.\n'
        '- "build_report": Generates markdown intelligence report.\n'
        '- "none": Direct navigation/conversational response without specialist execution.\n\n'
        "CURRENT APPLICATION STATE:\n"
        f"  has_image_1={has_image_1}, has_image_2={has_image_2}\n"
        f"  current_tab={current_tab}\n"
        f"  selected_cluster={selected_cluster}\n"
        f"  prior_analysis={prior_summary}\n\n"
        "FOLLOW-UP RULES:\n"
        "1. If user asks 'Which is biggest?', 'Show largest', 'Show second largest': set specialist='rank_change_clusters', reuse_prior=true.\n"
        "2. If user asks 'Show me' or 'Show the change map' or 'Highlight': set specialist='none', reuse_prior=true, add appropriate ui_actions.\n"
        "3. If user asks 'Where is it?' or 'Coordinates': if prior geospatial exists, set specialist='geospatial', reuse_prior=true.\n"
        "4. If user asks 'Why does this matter?' or 'Explain this': set specialist='none', reuse_prior=true, add 'show_why_this_matters' or 'show_why_this_result'.\n"
        "5. If user asks 'Executive summary' or 'Investigate this': set specialist='build_summary' or appropriate multi-step chain.\n"
        "6. If user asks for bi-temporal change but has_image_2 is false: set missing_input='image_2', requires_specialist=false.\n\n"
        "ALLOWED UI ACTIONS (verbatim strings for ui_actions list):\n"
        '  "navigate_tab:change-mask", "navigate_tab:interactive-overlay", "navigate_tab:side-by-side", "navigate_tab:diff-map",\n'
        '  "select_cluster:0", "highlight_cluster", "show_geospatial", "show_grounding", "show_optical_sar",\n'
        '  "open_trace", "open_evidence", "open_impact", "show_why_this_result", "show_why_this_matters",\n'
        '  "show_executive_summary", "generate_report", "speak_response"\n\n'
        "Return ONLY valid JSON:\n"
        '{\n'
        '  "intent": "brief explanation",\n'
        '  "goal": "investigate_change / follow_up / navigate / inspect_region",\n'
        '  "requires_specialist": true,\n'
        '  "specialist": "change_detection",\n'
        '  "question": "question for model if needed",\n'
        '  "target_cluster": 0,\n'
        '  "reuse_prior": false,\n'
        '  "missing_input": null,\n'
        '  "ui_actions": ["navigate_tab:change-mask", "select_cluster:0", "open_evidence"],\n'
        '  "response_style": "analytical"\n'
        '}'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    raw = await _call_qwen3(messages, timeout=120.0)
    try:
        plan = _extract_json_from_llm(raw)
        if plan.get("specialist") not in ALLOWED_SPECIALIST_TOOLS:
            plan["specialist"] = "none"
            plan["requires_specialist"] = False
        if not isinstance(plan.get("ui_actions"), list):
            plan["ui_actions"] = []
        return plan
    except Exception:
        # Graceful fallback heuristic
        q_lower = query.lower()
        if any(w in q_lower for w in ["biggest", "largest", "rank"]):
            return {
                "intent": "Identify largest change cluster", "goal": "rank_clusters",
                "requires_specialist": False, "specialist": "rank_change_clusters",
                "reuse_prior": True, "target_cluster": 0,
                "ui_actions": ["navigate_tab:interactive-overlay", "select_cluster:0", "highlight_cluster"],
                "response_style": "analytical"
            }
        elif any(w in q_lower for w in ["where", "coordinate", "location", "crs"]):
            return {
                "intent": "Inspect geographic location", "goal": "geospatial",
                "requires_specialist": False, "specialist": "geospatial",
                "reuse_prior": True,
                "ui_actions": ["show_geospatial", "open_evidence"],
                "response_style": "analytical"
            }
        elif any(w in q_lower for w in ["why", "matter", "impact", "significance"]):
            return {
                "intent": "Explain domain significance", "goal": "impact",
                "requires_specialist": False, "specialist": "none",
                "reuse_prior": True,
                "ui_actions": ["show_why_this_matters", "open_impact"],
                "response_style": "analytical"
            }
        elif any(w in q_lower for w in ["summary", "executive"]):
            return {
                "intent": "Generate executive summary", "goal": "summary",
                "requires_specialist": False, "specialist": "build_summary",
                "reuse_prior": True,
                "ui_actions": ["show_executive_summary", "open_summary"],
                "response_style": "analytical"
            }
        return {
            "intent": query, "goal": "general_query", "requires_specialist": False,
            "specialist": "none", "question": query, "reuse_prior": bool(prior_result),
            "ui_actions": [], "response_style": "conversational"
        }


async def _qwen3_synthesize(
    query: str,
    plan: dict,
    specialist_result: Optional[dict],
    prior_result: Optional[dict],
    selected_cluster_info: Optional[dict] = None
) -> dict:
    """Ask Qwen3 to synthesize a grounded response with provenance, why_this_result, why_this_matters, and executive summary."""
    result_to_use = specialist_result or prior_result

    # Format factual evidence
    if result_to_use:
        SKIP_KEYS = {
            "change_mask", "difference_map", "overlay_t1", "overlay_t2",
            "fusion_visualization", "visualization", "markdown_report"
        }
        safe = {
            k: v for k, v in result_to_use.items()
            if isinstance(v, (str, int, float, bool, list, dict))
            and k not in SKIP_KEYS
            and not k.endswith("_url")
        }
        if selected_cluster_info:
            safe["selected_cluster_detail"] = selected_cluster_info
        specialist_section = (
            "ACTUAL SPECIALIST EVIDENCE (FACTUAL VALUES — YOU MUST ONLY REFERENCE THESE):\n"
            + _json.dumps(safe, indent=2, default=str)[:3500]
        )
    else:
        specialist_section = (
            "No specialist analysis has run or no imagery was loaded. "
            "Answer honestly from available context. Do not invent any numbers or detections."
        )

    system_prompt = (
        "You are JARVIS 2.0, the AI Aerospace & Remote Sensing Analyst for SatQuery AI.\n\n"
        "STRICT ANTI-HALLUCINATION & HONESTY CONTRACT:\n"
        "1. NEVER fabricate numbers, percentages, pixel counts, bounding boxes, or confidence values.\n"
        "2. Reference ONLY values present in the ACTUAL SPECIALIST EVIDENCE.\n"
        "3. Clearly distinguish MODEL CONFIDENCE from GROUND TRUTH VALIDATION.\n"
        "4. speak_text must be 1-2 natural conversational sentences (<25 words total), natural and concise.\n"
        "5. why_this_matters must use hedged, professional analyst phrasing ('could indicate', 'may be relevant to', 'potential application in').\n"
        "6. If no geospatial metadata exists, state that coordinates are unavailable; never invent coordinates.\n"
        "7. If images are missing, state what image is needed.\n\n"
        "Return ONLY valid JSON:\n"
        '{\n'
        '  "answer": "Detailed technical analysis referencing exact values",\n'
        '  "speak_text": "Short conversational voice response under 25 words",\n'
        '  "why_this_result": {\n'
        '    "model": "Model name that produced the result",\n'
        '    "inputs": "Input imagery used",\n'
        '    "output_meaning": "What the metric/mask represents",\n'
        '    "confidence_type": "Model confidence (not ground truth)",\n'
        '    "limitations": "Model domain and boundary limits"\n'
        '  },\n'
        '  "why_this_matters": "Carefully hedged domain significance (urban expansion, infrastructure, land-use)",\n'
        '  "recommended_next_step": "Evidence-based recommendation for the analyst",\n'
        '  "executive_summary": {\n'
        '    "finding": "Primary finding",\n'
        '    "largest_change": "Largest detected region info",\n'
        '    "evidence": "Key specialist metrics",\n'
        '    "geospatial_info": "Geospatial status/metadata",\n'
        '    "potential_relevance": "Potential application",\n'
        '    "limitations": "Scientific limitations"\n'
        '  }\n'
        '}'
    )

    user_msg = (
        f'User Query: "{query}"\n'
        f"Intent/Plan Goal: {plan.get('goal', 'analysis')}\n\n"
        f"{specialist_section}\n\n"
        "Synthesize grounded analyst output following the strict rules."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]
    raw = await _call_qwen3(messages, timeout=150.0)
    try:
        synth = _extract_json_from_llm(raw)
        # Ensure speak_text is speech-normalized
        if "speak_text" in synth and synth["speak_text"]:
            synth["speak_text"] = prepare_speech_text(synth["speak_text"])
        return synth
    except Exception:
        fallback_answer = (
            result_to_use.get("answer", "Analysis complete.") if result_to_use
            else "Please ensure imagery is loaded to run analysis."
        )
        return {
            "answer": fallback_answer,
            "speak_text": prepare_speech_text(fallback_answer),
            "why_this_result": {
                "model": result_to_use.get("task", "Specialist Engine") if result_to_use else "None",
                "inputs": "T1/T2 Remote Sensing Imagery",
                "output_meaning": "Detected spatial difference features",
                "confidence_type": "Model Output Confidence",
                "limitations": "Requires ground validation for semantic confirmation"
            },
            "why_this_matters": "Detected spatial differences could be relevant to infrastructure or land-use monitoring.",
            "recommended_next_step": "Review the largest detected change cluster on the interactive overlay.",
            "executive_summary": {
                "finding": fallback_answer,
                "largest_change": "Cluster 0",
                "evidence": str(result_to_use.get("change_percentage", "N/A")) if result_to_use else "N/A",
                "geospatial_info": "Standard pixel coordinates",
                "potential_relevance": "Land cover & infrastructure monitoring",
                "limitations": "Validation required"
            }
        }


def _build_jarvis_ui_actions(
    plan: dict,
    specialist_result: Optional[dict],
    trace_steps: list,
    target_cluster: Optional[int] = None
) -> list:
    """Build a VALIDATED list of UI actions — strict adherence to ALLOWED_UI_ACTIONS."""
    actions: list = []

    if trace_steps:
        actions.append({"action": "show_agent_progress", "steps": trace_steps})

    if specialist_result:
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


async def _execute_jarvis_specialist(
    specialist: str, req: JarvisRequest, plan: dict
) -> Tuple[dict, str]:
    """Execute the selected specialist model and return (formatted_result, task_id)."""
    question = plan.get("question", "") or req.query

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
    # 1. Strict verification: Ollama qwen3:8b
    ollama_ok, ollama_reason = await _check_ollama_status()
    if not ollama_ok:
        raise HTTPException(
            status_code=503,
            detail=f"JARVIS is unavailable because Ollama {OLLAMA_MODEL} could not be reached. {ollama_reason}"
        )

    # 2. Retrieve cached prior result if provided
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
        # 3. State-aware Qwen3:8b Planning
        trace_steps.append(f"Building analysis plan with {OLLAMA_MODEL}")
        plan = await _qwen3_plan(
            query=req.query,
            has_image_1=req.has_image_1,
            has_image_2=req.has_image_2,
            current_tab=req.current_tab or "overview",
            prior_result=prior_result,
            selected_cluster=req.selected_cluster
        )
        specialist = plan.get("specialist", "none")
        goal = plan.get("goal", "analysis")
        trace_steps.append(f"Plan goal: {goal}")
        trace_steps.append(f"Specialist selected: {specialist}")

        # 4. Handle Missing Requirements honestly
        if plan.get("missing_input") == "image_2" or (specialist == "change_detection" and not req.has_image_2):
            missing_text = (
                "I have the first image, but I need the second acquisition to perform a bi-temporal comparison."
                if req.has_image_1 else "Please upload or select the imagery pair to run bi-temporal change detection."
            )
            audio_info = await tts_service.synthesize_base64(missing_text) if req.voice_enabled else {}
            return {
                "answer": missing_text,
                "speak_text": prepare_speech_text(missing_text),
                "audio_base64": audio_info.get("audio_base64"),
                "evidence": {},
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

        if specialist in ("rank_change_clusters", "inspect_change_cluster") or (plan.get("reuse_prior") and prior_result):
            specialist_result = prior_result
            new_task_id = req.prior_result_id
            trace_steps.append("Reusing cached specialist analysis (avoided model rerun)")
            if prior_result and prior_result.get("bboxes"):
                ranked = ClusterAnalytics.rank_clusters(prior_result["bboxes"])
                selected_cluster_info = ClusterAnalytics.get_cluster(ranked, target_cluster_idx)
                trace_steps.append(f"Ranked {len(ranked)} clusters; selected cluster {target_cluster_idx}")

        elif plan.get("requires_specialist") and specialist != "none":
            if not req.has_image_1:
                raise HTTPException(400, detail="No imagery loaded. Please upload or select images first.")
            trace_steps.append(f"Executing {specialist} specialist")
            try:
                specialist_result, new_task_id = await _execute_jarvis_specialist(specialist, req, plan)
                et = specialist_result.get("execution_time_sec", 0) or 0
                trace_steps.append(f"Specialist inference complete ({et:.1f}s)")
                if specialist_result.get("bboxes"):
                    ranked = ClusterAnalytics.rank_clusters(specialist_result["bboxes"])
                    selected_cluster_info = ClusterAnalytics.get_cluster(ranked, 0)
                    trace_steps.append(f"Identified {len(ranked)} change clusters")
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(500, detail=f"Specialist {specialist} failed: {exc}")
        else:
            trace_steps.append("Contextual synthesis (no specialist execution required)")

        # 6. Synthesize Grounded Analyst Response with Qwen3:8b
        trace_steps.append(f"Synthesizing grounded analysis with {OLLAMA_MODEL}")
        synthesis = await _qwen3_synthesize(
            query=req.query,
            plan=plan,
            specialist_result=specialist_result,
            prior_result=prior_result,
            selected_cluster_info=selected_cluster_info
        )

        # 7. Generate Validated UI Actions
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

        return {
            "answer": synthesis.get("answer", "Analysis complete."),
            "speak_text": speak_text,
            "audio_base64": audio_base64,
            "evidence": specialist_result or {},
            "provenance": provenance,
            "why_this_result": synthesis.get("why_this_result", {}),
            "why_this_matters": synthesis.get("why_this_matters", ""),
            "recommended_next_step": synthesis.get("recommended_next_step", ""),
            "executive_summary": synthesis.get("executive_summary", {}),
            "ui_actions": ui_actions,
            "task_id": jarvis_task_id,
            "specialist_task_id": new_task_id,
            "model": OLLAMA_MODEL,
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
