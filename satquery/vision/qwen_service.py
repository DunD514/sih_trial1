"""
Dedicated Vision Service for Qwen2-VL-2B-Instruct in SatQuery AI.
Extracts structured visual context from satellite and remote sensing imagery.
Supports local weights, CUDA with CPU fallback, image preprocessing, structured parsing,
in-memory caching, and semantic consistency validation.
"""

import os
import re
import json
import time
import hashlib
import logging
from collections import OrderedDict
from threading import Lock
from typing import Dict, Any, Optional, Union, List
from io import BytesIO

import torch
from PIL import Image

logger = logging.getLogger("satquery.vision.qwen")


def resolve_default_model_path() -> str:
    env_path = os.getenv("QWEN_MODEL_PATH")
    if env_path:
        return env_path
    local_candidates = [
        os.path.abspath("Qwen2-VL-2B-Instruct"),
        os.path.join(os.getcwd(), "Qwen2-VL-2B-Instruct"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Qwen2-VL-2B-Instruct")),
        "Qwen2-VL-2B-Instruct"
    ]
    for p in local_candidates:
        if os.path.isdir(p) and (
            os.path.isfile(os.path.join(p, "config.json")) or
            os.path.isfile(os.path.join(p, "model.safetensors.index.json"))
        ):
            return os.path.abspath(p)
    return "Qwen/Qwen2-VL-2B-Instruct"


# Environment configurations
QWEN_MODEL_PATH = resolve_default_model_path()
QWEN_DEVICE = os.getenv("QWEN_DEVICE", "auto")
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "256"))

# Standard 20-field logical schema for structured visual context
EMPTY_VISUAL_CONTEXT: Dict[str, Any] = {
    "image_type": "optical_satellite",
    "summary": "",
    "objects": [],
    "land_cover": [],
    "land_use": [],
    "infrastructure": [],
    "terrain": [],
    "vegetation": [],
    "water_features": [],
    "agriculture": [],
    "urban_features": [],
    "industrial_features": [],
    "text_detected": [],
    "spatial_relationships": [],
    "notable_conditions": [],
    "possible_changes": [],
    "observations": [],
    "inferences": [],
    "confidence": "moderate",
    "uncertainties": []
}

# Optimized concise prompt for fast, accurate structured extraction
QWEN_VISION_PROMPT = """You are a remote-sensing vision analyst. Analyze the supplied satellite/aerial image.
Identify observable visual features: buildings, roads, vegetation, water, bare land, agricultural plots, terrain, and spatial layout.
Distinguish direct visual observations from interpretations. Do not hallucinate coordinates, dates, or sensor names.

Return ONLY a valid JSON object adhering to this schema:
{
  "image_type": "string (e.g. optical_satellite, aerial, SAR)",
  "summary": "string (concise factual overview of the image)",
  "objects": ["detected visual objects (e.g. buildings, roads, trees)"],
  "land_cover": ["land cover types (e.g. vegetation, built-up, bare soil, water)"],
  "land_use": ["land use categories (e.g. residential, agricultural, industrial)"],
  "infrastructure": ["infrastructure elements (e.g. roads, roofs, utilities)"],
  "terrain": ["terrain types (e.g. flat, hilly, coastal)"],
  "vegetation": ["vegetation features (e.g. dense trees, cropland, sparse grass, none)"],
  "water_features": ["water bodies (e.g. rivers, canals, none)"],
  "agriculture": ["agricultural features (e.g. cultivated fields, none)"],
  "urban_features": ["urban layout features (e.g. building clusters, none)"],
  "industrial_features": ["industrial structures (e.g. warehouses, none)"],
  "text_detected": [],
  "spatial_relationships": ["spatial arrangements of features"],
  "notable_conditions": ["notable conditions (e.g. cleared plots, none)"],
  "possible_changes": ["possible change indicators (e.g. new construction, none)"],
  "observations": ["direct factual visual observations"],
  "inferences": ["interpretive deductions"],
  "confidence": "high, moderate, or low",
  "uncertainties": ["uncertainties or limitations"]
}"""


class VisualContextCache:
    """
    Thread-safe, bounded in-memory LRU cache with JSON disk persistence for structured visual context.
    Caches Qwen visual context keyed by SHA-256 hash of image content and optional metadata.
    Avoids redundant expensive Qwen inference for repeated questions on the same image.
    """

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._disk_path = os.path.join(os.path.dirname(__file__), "visual_cache_store.json")
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if os.path.isfile(self._disk_path):
            try:
                with open(self._disk_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    with self._lock:
                        for k, v in data.items():
                            self._cache[k] = v
                logger.info("Loaded %d visual context items from persistent store: %s", len(self._cache), self._disk_path)
            except Exception as e:
                logger.warning("Could not load persistent visual cache from %s: %s", self._disk_path, e)

    def _save_to_disk(self) -> None:
        try:
            with open(self._disk_path, "w", encoding="utf-8") as f:
                json.dump(dict(self._cache), f, indent=2)
        except Exception as e:
            logger.warning("Could not persist visual cache to %s: %s", self._disk_path, e)

    @staticmethod
    def compute_key(image_input: Union[str, Image.Image, bytes], metadata: Optional[Dict[str, Any]] = None) -> str:
        hasher = hashlib.sha256()

        if isinstance(image_input, str):
            if os.path.isfile(image_input):
                with open(image_input, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
            else:
                hasher.update(image_input.encode("utf-8"))
        elif isinstance(image_input, bytes):
            hasher.update(image_input)
        elif isinstance(image_input, Image.Image):
            hasher.update(image_input.tobytes())
            hasher.update(f"{image_input.size}_{image_input.mode}".encode("utf-8"))
        else:
            hasher.update(str(image_input).encode("utf-8"))

        if metadata:
            meaningful_meta = {k: v for k, v in metadata.items() if v}
            if meaningful_meta:
                hasher.update(json.dumps(meaningful_meta, sort_keys=True).encode("utf-8"))

        return hasher.hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return json.loads(json.dumps(self._cache[key]))
            return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = json.loads(json.dumps(value))
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
            self._save_to_disk()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._save_to_disk()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class ImagePreprocessor:
    """
    Safely validates, cleans, and prepares satellite/aerial images for Qwen2-VL.
    Preserves aspect ratio, spatial resolution, and prevents out-of-memory errors.
    """

    MAX_DIMENSION = 1280
    MIN_DIMENSION = 224

    @classmethod
    def preprocess(cls, image_input: Union[str, Image.Image, bytes]) -> Image.Image:
        """
        Loads and prepares an image for vision inference.
        """
        img: Optional[Image.Image] = None

        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Image path not found: {image_input}")
            img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Validate format & convert safely to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Check dimensions & resize only if excessively large, strictly preserving aspect ratio
        w, h = img.size
        if max(w, h) > cls.MAX_DIMENSION:
            scale = cls.MAX_DIMENSION / max(w, h)
            new_w = max(cls.MIN_DIMENSION, int(w * scale))
            new_h = max(cls.MIN_DIMENSION, int(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            logger.info("Resized image for Qwen2-VL from (%dx%d) to (%dx%d)", w, h, new_w, new_h)

        return img


class SemanticConsistencyValidator:
    """
    Deterministic post-processor to detect, resolve, or flag semantic contradictions
    between fields in the extracted VisualContext.
    """

    @classmethod
    def validate_and_normalize(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(context)
        uncertainties = list(result.get("uncertainties", []))

        # Helper to sanitize list fields: remove "none" if actual elements exist
        def clean_list(items: Any) -> List[str]:
            if not isinstance(items, list):
                if items and str(items).strip().lower() not in ("none", "null", "[]"):
                    return [str(items).strip()]
                return []
            cleaned = []
            for item in items:
                s = str(item).strip()
                if s and s not in cleaned:
                    cleaned.append(s)
            if len(cleaned) > 1:
                cleaned = [x for x in cleaned if x.lower() != "none"]
            return cleaned

        for k in (
            "objects", "land_cover", "land_use", "infrastructure", "terrain",
            "vegetation", "water_features", "agriculture", "urban_features",
            "industrial_features", "text_detected", "spatial_relationships",
            "notable_conditions", "possible_changes", "observations", "inferences"
        ):
            result[k] = clean_list(result.get(k, []))

        observations_text = " ".join(result["observations"]).lower()
        land_cover = [x.lower() for x in result["land_cover"]]
        objects = result["objects"]

        # 1. Water contradiction check
        has_water_lc = any("water" in lc for lc in land_cover)
        water_feats = [wf.lower() for wf in result["water_features"]]
        water_in_obs = "water" in observations_text or "river" in observations_text or "lake" in observations_text

        if (has_water_lc or water_in_obs) and ("none" in water_feats or not water_feats):
            if water_in_obs:
                result["water_features"] = ["water body (observed)"]
            else:
                result["water_features"] = ["water body (inferred from land cover)"]
                uncertainties.append("Water identified in land cover but specific water feature details are limited.")
        elif not has_water_lc and not water_in_obs and water_feats and water_feats != ["none"]:
            uncertainties.append("Water feature reported without corresponding water land cover classification.")

        # 2. Built-up / Urban contradiction check
        has_built_lc = any("built" in lc or "urban" in lc for lc in land_cover)
        urban_feats = [uf.lower() for uf in result["urban_features"]]
        infra_feats = [inf.lower() for inf in result["infrastructure"]]
        built_in_obs = "building" in observations_text or "structure" in observations_text or "roof" in observations_text or "road" in observations_text

        if (has_built_lc or built_in_obs) and ("none" in urban_feats or not urban_feats):
            if built_in_obs:
                result["urban_features"] = ["structures (observed)"]
            elif has_built_lc:
                result["urban_features"] = ["built structures (inferred from land cover)"]

        # 3. Synchronize observed structures into objects list if objects is empty
        if not objects and built_in_obs:
            detected_obs_objs = []
            if "building" in observations_text or "structure" in observations_text or "roof" in observations_text:
                detected_obs_objs.append("buildings")
            if "road" in observations_text:
                detected_obs_objs.append("roads")
            if detected_obs_objs:
                result["objects"] = detected_obs_objs

        # 4. Vegetation contradiction check
        has_veg_lc = any("veg" in lc or "crop" in lc or "forest" in lc or "grass" in lc for lc in land_cover)
        veg_feats = [vf.lower() for vf in result["vegetation"]]
        if has_veg_lc and ("none" in veg_feats or not veg_feats):
            result["vegetation"] = ["vegetation cover"]

        # 5. Evaluate Confidence dynamically based on visual evidence
        obs_count = len(result["observations"])
        obj_count = len(result["objects"])
        unc_count = len(uncertainties)
        is_fallback = any("fallback" in str(u).lower() for u in uncertainties)

        if is_fallback:
            computed_conf = "low"
        elif (obs_count >= 2 and obj_count >= 1 and unc_count <= 1) or (obs_count >= 1 and obj_count >= 1 and unc_count == 0):
            computed_conf = "high"
        elif obs_count >= 1 or obj_count >= 1 or len(result["land_cover"]) >= 1:
            computed_conf = "moderate"
        else:
            computed_conf = "low"

        # Downgrade if multiple uncertainties
        if unc_count >= 3 and computed_conf == "high":
            computed_conf = "moderate"

        result["confidence"] = computed_conf
        result["uncertainties"] = clean_list(uncertainties)

        return result


class VisualContextParser:
    """
    Parses, validates, and normalizes Qwen2-VL output into structured JSON
    adhering to the standard 20-field schema.
    """

    @classmethod
    def parse(cls, raw_output: str) -> Dict[str, Any]:
        """
        Extracts JSON from text and normalizes into the strict 20-field schema.
        """
        if not raw_output or not raw_output.strip():
            res = dict(EMPTY_VISUAL_CONTEXT)
            res["summary"] = "No visual output generated"
            res["confidence"] = "low"
            return res

        cleaned = raw_output.strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        # Repair list elements formatted as "key": "val" inside array brackets
        repaired = re.sub(r'\[\s*"[^"]+"\s*:\s*("[^"]+")', r'[\1', cleaned)
        repaired = re.sub(r',\s*"[^"]+"\s*:\s*("[^"]+")', r', \1', repaired)

        parsed: Optional[Dict[str, Any]] = None
        for candidate in (cleaned, repaired):
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                m = re.search(r"\{[\s\S]*\}", candidate)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        break
                    except json.JSONDecodeError:
                        pass

        if not isinstance(parsed, dict):
            parsed = dict(EMPTY_VISUAL_CONTEXT)
            for key in EMPTY_VISUAL_CONTEXT:
                pattern = rf'"{key}"\s*:\s*(\[[^\]]*\]|"[^"]*")'
                m = re.search(pattern, cleaned)
                if m:
                    val_str = m.group(1).strip()
                    try:
                        parsed[key] = json.loads(val_str)
                    except json.JSONDecodeError:
                        if val_str.startswith('"') and val_str.endswith('"'):
                            parsed[key] = val_str[1:-1]
                        elif val_str.startswith('['):
                            items = [x.strip(' "\'') for x in re.findall(r'"([^"]+)"', val_str)]
                            parsed[key] = items

            if not parsed.get("summary"):
                parsed["summary"] = cleaned[:300].replace("\n", " ").strip()
            
            raw_obs = [line.strip("- * \t\r") for line in cleaned.split("\n") if line.strip()]
            valid_obs = [o for o in raw_obs if not o.startswith(("{", "}", "[", "]", '"', "'", "image_type", "summary"))]
            if not parsed.get("observations") and valid_obs:
                parsed["observations"] = valid_obs[:5]

            parsed["confidence"] = "low"
            parsed.setdefault("uncertainties", []).append("Structured JSON parsing fallback used")
            return SemanticConsistencyValidator.validate_and_normalize(parsed)

        # Merge with default schema to ensure all 20 fields are present
        normalized = dict(EMPTY_VISUAL_CONTEXT)
        for key in EMPTY_VISUAL_CONTEXT:
            if key in parsed:
                val = parsed[key]
                if isinstance(EMPTY_VISUAL_CONTEXT[key], list):
                    if not isinstance(val, list):
                        val = [str(val)] if val else []
                    else:
                        # Extract readable strings if elements are dicts
                        cleaned_elements = []
                        for elem in val:
                            if isinstance(elem, dict):
                                desc = elem.get("description") or elem.get("label") or elem.get("name") or elem.get("type")
                                if desc:
                                    cleaned_elements.append(str(desc))
                                else:
                                    cleaned_elements.append(json.dumps(elem))
                            elif elem is not None:
                                cleaned_elements.append(str(elem))
                        val = cleaned_elements
                normalized[key] = val

        return SemanticConsistencyValidator.validate_and_normalize(normalized)


class Qwen2VLService:
    """
    Dedicated singleton service managing the local Qwen2-VL-2B-Instruct vision model.
    Loads model once into GPU/CPU memory, caches results, and executes inference.
    """

    _instance: Optional["Qwen2VLService"] = None

    def __init__(
        self,
        model_path: str = QWEN_MODEL_PATH,
        device_setting: str = QWEN_DEVICE,
        max_new_tokens: int = QWEN_MAX_NEW_TOKENS
    ):
        self.model_path = model_path
        self.device_setting = device_setting
        self.max_new_tokens = max_new_tokens

        self._model = None
        self._processor = None
        self._device = None
        self._load_error: Optional[str] = None
        self._is_initialized = False
        self.cache = VisualContextCache()

    @classmethod
    def get_instance(cls) -> "Qwen2VLService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _determine_device(self) -> torch.device:
        if self.device_setting.lower() == "cpu":
            return torch.device("cpu")
        elif self.device_setting.lower() == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        else:  # auto
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> bool:
        """
        Loads Qwen2-VL model and processor once.
        Returns True if successful, False otherwise.
        """
        if self._is_initialized and self._model is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            logger.info("Initializing Qwen2-VL-2B-Instruct from: %s", self.model_path)
            self._device = self._determine_device()
            logger.info("Selected hardware device for Qwen2-VL: %s", self._device)

            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            torch_dtype = torch.float16 if self._device.type == "cuda" else torch.float32
            is_local = os.path.isdir(self.model_path)

            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=is_local,
                trust_remote_code=True
            )

            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map=None,
                local_files_only=is_local,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self._device)

            self._model.eval()
            self._is_initialized = True
            logger.info("Qwen2-VL-2B-Instruct successfully loaded and ready on %s.", self._device)
            return True

        except Exception as exc:
            self._load_error = str(exc)
            logger.error("Failed to load Qwen2-VL model: %s", exc)
            return False

    def is_available(self) -> bool:
        return self._is_initialized and self._model is not None

    def analyze_image(
        self,
        image_input: Union[str, Image.Image, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        custom_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Executes vision context extraction for a remote sensing image with cache support.
        Returns a dictionary containing status, visual_context, metadata, and cache status.
        """
        start_time = time.time()

        # 1. Check in-memory cache
        cache_key: Optional[str] = None
        if use_cache:
            try:
                cache_key = VisualContextCache.compute_key(image_input, metadata)
                cached_context = self.cache.get(cache_key)
                if cached_context is not None:
                    duration = time.time() - start_time
                    logger.info("Visual context cache HIT for key %s (%.4fs)", cache_key[:12], duration)
                    return {
                        "status": "completed",
                        "visual_context": cached_context,
                        "raw_output": json.dumps(cached_context),
                        "model_name": "Qwen2-VL-2B-Instruct",
                        "device": str(self._device or "cuda"),
                        "inference_time_sec": duration,
                        "cached": True
                    }
            except Exception as e:
                logger.warning("Cache key computation error: %s", e)

        # 2. Preprocess Image
        try:
            pil_img = ImagePreprocessor.preprocess(image_input)
        except Exception as e:
            logger.error("Image preprocessing error: %s", e)
            return {
                "status": "error",
                "visual_context": None,
                "vision_error": f"Image preprocessing failed: {str(e)}",
                "inference_time_sec": time.time() - start_time,
                "cached": False
            }

        # 3. Ensure Model is Loaded
        if not self._is_initialized:
            loaded = self.load_model()
            if not loaded:
                return {
                    "status": "partial",
                    "visual_context": None,
                    "vision_error": f"Qwen2-VL model unavailable: {self._load_error}",
                    "inference_time_sec": time.time() - start_time,
                    "cached": False
                }

        # 4. Build Prompt & Format Inputs
        prompt_text = custom_prompt or QWEN_VISION_PROMPT
        if metadata:
            meta_str = f"\nAdditional metadata provided: {json.dumps(metadata)}"
            prompt_text += meta_str

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        try:
            text_prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text_prompt],
                images=[pil_img],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self._device)

            logger.info("Starting optimized Qwen2-VL inference on %s...", self._device)
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True
                )

            # Trim input prompt tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            duration = time.time() - start_time
            logger.info("Qwen2-VL inference completed in %.2fs", duration)

            # 5. Parse Structured Context & Apply Semantic Validation
            visual_context = VisualContextParser.parse(output_text)

            # 6. Store in cache
            if use_cache and cache_key and visual_context:
                self.cache.put(cache_key, visual_context)

            return {
                "status": "completed",
                "visual_context": visual_context,
                "raw_output": output_text,
                "model_name": "Qwen2-VL-2B-Instruct",
                "device": str(self._device),
                "inference_time_sec": duration,
                "cached": False
            }

        except Exception as exc:
            duration = time.time() - start_time
            logger.error("Qwen2-VL inference failed: %s", exc)
            return {
                "status": "partial",
                "visual_context": None,
                "vision_error": f"Qwen2-VL inference error: {str(exc)}",
                "inference_time_sec": duration,
                "cached": False
            }


# Global singleton instance
qwen_vision_service = Qwen2VLService.get_instance()
