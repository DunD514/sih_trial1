"""
Comprehensive Unit Tests for Qwen2-VL Integration & Ollama Coexistence.
Validates ImagePreprocessor, 20-field VisualContext schema, AnalysisOrchestrator prompt fusion,
Ollama co-execution, SceneContext synchronization, AgentController integration, and graceful error fallbacks.
"""

import os
import pytest
import numpy as np
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch

from satquery.vision.qwen_service import (
    ImagePreprocessor,
    VisualContextParser,
    VisualContextCache,
    SemanticConsistencyValidator,
    Qwen2VLService,
    EMPTY_VISUAL_CONTEXT,
    QWEN_MODEL_PATH,
    QWEN_DEVICE
)
from satquery.vision.orchestrator import AnalysisOrchestrator
from satquery.agent.context import SceneContext
from satquery.agent.controller import AgentController


class TestImagePreprocessor:
    """Validates remote sensing image preparation for Qwen2-VL."""

    def test_preprocess_pil_rgb(self):
        img = Image.new("RGB", (400, 300), color=(100, 150, 200))
        processed = ImagePreprocessor.preprocess(img)
        assert isinstance(processed, Image.Image)
        assert processed.mode == "RGB"
        assert processed.size == (400, 300)

    def test_preprocess_rgba_conversion(self):
        img = Image.new("RGBA", (500, 500), color=(50, 100, 150, 255))
        processed = ImagePreprocessor.preprocess(img)
        assert processed.mode == "RGB"
        assert processed.size == (500, 500)

    def test_preprocess_resize_large_image(self):
        # 3000 x 1500 (aspect ratio 2:1)
        img = Image.new("RGB", (3000, 1500), color=(0, 0, 0))
        processed = ImagePreprocessor.preprocess(img)
        assert max(processed.size) <= ImagePreprocessor.MAX_DIMENSION
        # Verify aspect ratio preserved (approx 2:1)
        w, h = processed.size
        assert abs((w / h) - 2.0) < 0.05

    def test_preprocess_missing_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            ImagePreprocessor.preprocess("non_existent_file_path_xyz.png")

    def test_preprocess_real_levir_sample(self):
        real_path = os.path.join("samples_LEVIR", "A", "test_2_0000_0000.png")
        if os.path.isfile(real_path):
            processed = ImagePreprocessor.preprocess(real_path)
            assert isinstance(processed, Image.Image)
            assert processed.mode == "RGB"
            assert processed.size[0] > 0 and processed.size[1] > 0


class TestVisualContextParser:
    """Validates the strict 20-field structured schema enforcement and parsing."""

    def test_all_20_fields_present_in_empty_schema(self):
        expected_fields = [
            "image_type", "summary", "objects", "land_cover", "land_use",
            "infrastructure", "terrain", "vegetation", "water_features",
            "agriculture", "urban_features", "industrial_features",
            "text_detected", "spatial_relationships", "notable_conditions",
            "possible_changes", "observations", "inferences", "confidence",
            "uncertainties"
        ]
        assert len(expected_fields) == 20
        assert len(EMPTY_VISUAL_CONTEXT) == 20
        for field in expected_fields:
            assert field in EMPTY_VISUAL_CONTEXT, f"Missing field in schema: {field}"

    def test_parse_valid_json_20_fields(self):
        raw_json = """
        {
            "image_type": "optical_satellite",
            "summary": "Suburban settlement with dense residential buildings and road grid.",
            "objects": ["buildings", "roads", "trees"],
            "land_cover": ["built-up", "vegetation", "bare soil"],
            "land_use": ["residential"],
            "infrastructure": ["asphalt roads", "rooftops"],
            "terrain": ["flat"],
            "vegetation": ["sparse trees"],
            "water_features": ["none"],
            "agriculture": ["none"],
            "urban_features": ["grid street pattern"],
            "industrial_features": ["none"],
            "text_detected": [],
            "spatial_relationships": ["buildings arranged along north-south roads"],
            "notable_conditions": ["active construction on eastern perimeter"],
            "possible_changes": ["ground clearing"],
            "observations": ["high rooftop density", "cleared rectangular plot"],
            "inferences": ["potential residential expansion"],
            "confidence": "high",
            "uncertainties": ["crop type in adjacent plot"]
        }
        """
        parsed = VisualContextParser.parse(raw_json)
        assert len(parsed) == 20
        assert parsed["image_type"] == "optical_satellite"
        assert "Suburban" in parsed["summary"]
        assert len(parsed["land_cover"]) == 3
        assert "asphalt roads" in parsed["infrastructure"]
        assert parsed["confidence"] == "high"
        assert len(parsed["observations"]) == 2

    def test_parse_markdown_fenced_json(self):
        fenced = """```json
        {
            "image_type": "aerial",
            "summary": "Agricultural plots and river channel.",
            "water_features": ["river channel"],
            "confidence": "moderate"
        }
        ```"""
        parsed = VisualContextParser.parse(fenced)
        assert len(parsed) == 20
        assert parsed["image_type"] == "aerial"
        assert parsed["water_features"] == ["river channel"]
        assert isinstance(parsed["land_cover"], list)
        assert isinstance(parsed["uncertainties"], list)

    def test_parse_fallback_text(self):
        raw_text = "This scene shows a coastal harbor with several cargo vessels and warehouses."
        parsed = VisualContextParser.parse(raw_text)
        assert isinstance(parsed, dict)
        assert len(parsed) == 20
        assert "coastal harbor" in parsed["summary"]
        assert parsed["confidence"] == "low"
        assert "Structured JSON parsing fallback used" in parsed["uncertainties"]


class TestAnalysisOrchestrator:
    """Validates context fusion, prompt preparation, and Ollama integration."""

    def test_format_visual_context_block(self):
        vc = {
            "image_type": "optical_satellite",
            "summary": "Agricultural expanse with irrigation channels.",
            "objects": ["fields", "canals"],
            "land_cover": ["cropland", "water"],
            "land_use": ["agriculture"],
            "infrastructure": ["canals", "unpaved roads"],
            "terrain": ["flat plains"],
            "vegetation": ["dense crops"],
            "water_features": ["irrigation canal"],
            "agriculture": ["cultivated plots"],
            "urban_features": ["none"],
            "industrial_features": ["none"],
            "text_detected": [],
            "spatial_relationships": ["canals bordering fields"],
            "notable_conditions": ["active irrigation"],
            "possible_changes": ["none"],
            "observations": ["green crop canopy"],
            "inferences": ["cultivated agricultural zone"],
            "confidence": "high",
            "uncertainties": ["exact crop species"]
        }
        block = AnalysisOrchestrator.format_visual_context_block(vc)
        assert "=== VISUAL CONTEXT FROM QWEN2-VL ===" in block
        assert "=== END VISUAL CONTEXT ===" in block
        assert "Agricultural expanse" in block
        assert "cropland" in block

    def test_fuse_context_for_ollama(self):
        orch = AnalysisOrchestrator()
        base_prompt = "Analyze the changes between T1 and T2 acquisitions."
        vc = {"summary": "Urban structures along highway", "confidence": "high"}
        metadata = {"sensor": "Sentinel-2", "resolution_m": 10}
        analysis_data = {"change_percentage": 4.5, "clusters": 3}

        fused = orch.fuse_context_for_ollama(
            base_prompt=base_prompt,
            visual_context=vc,
            image_metadata=metadata,
            existing_analysis_data=analysis_data
        )

        assert base_prompt in fused
        assert "=== VISUAL CONTEXT FROM QWEN2-VL ===" in fused
        assert "=== IMAGE METADATA ===" in fused
        assert "Sentinel-2" in fused
        assert "=== REMOTE SENSING SPECIALIST EVIDENCE ===" in fused
        assert "4.5" in fused

    def test_ollama_reasoning_execution(self):
        import asyncio
        mock_ollama = MagicMock()
        mock_ollama.model = "qwen3:8b"
        mock_ollama.chat = AsyncMock(return_value="The visual evidence indicates expansion of infrastructure.")

        orch = AnalysisOrchestrator(ollama=mock_ollama)
        fused_prompt = "=== VISUAL CONTEXT FROM QWEN2-VL ===\nSummary: Built-up structures"
        
        result = asyncio.run(orch.execute_reasoning_with_ollama(fused_prompt))
        assert result["status"] == "success"
        assert "expansion of infrastructure" in result["analysis"]
        assert result["model"] == "qwen3:8b"
        mock_ollama.chat.assert_called_once()

    def test_graceful_handling_when_qwen_fails(self):
        mock_vision = MagicMock()
        mock_vision.analyze_image.side_effect = RuntimeError("CUDA Out of Memory simulation")

        orch = AnalysisOrchestrator(vision_service=mock_vision)
        img = Image.new("RGB", (100, 100))
        result = orch.extract_visual_context(img)

        # Must not crash, should return status partial
        assert result["status"] == "partial"
        assert result["visual_context"] is None
        assert "CUDA Out of Memory" in result["vision_error"]


class TestVisualContextCache:
    """Validates bounded LRU cache for visual context."""

    def test_cache_put_get(self):
        cache = VisualContextCache(max_size=5)
        key = VisualContextCache.compute_key("fake_img_1")
        vc = {"summary": "Test image 1", "confidence": "high"}
        
        cache.put(key, vc)
        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved["summary"] == "Test image 1"

    def test_cache_miss(self):
        cache = VisualContextCache(max_size=5)
        key = VisualContextCache.compute_key("missing_img")
        assert cache.get(key) is None

    def test_cache_lru_eviction(self):
        cache = VisualContextCache(max_size=2)
        k1 = VisualContextCache.compute_key("img_1")
        k2 = VisualContextCache.compute_key("img_2")
        k3 = VisualContextCache.compute_key("img_3")

        cache.put(k1, {"id": 1})
        cache.put(k2, {"id": 2})
        assert cache.size() == 2
        
        # Access k1 to make it recently used
        cache.get(k1)
        
        # Insert k3, k2 should be evicted
        cache.put(k3, {"id": 3})
        assert cache.size() == 2
        assert cache.get(k1) is not None
        assert cache.get(k2) is None
        assert cache.get(k3) is not None


class TestSemanticConsistencyValidator:
    """Validates contradiction detection, resolution, and confidence calculation."""

    def test_water_contradiction_resolution(self):
        raw = {
            "land_cover": ["vegetation", "water"],
            "water_features": ["none"],
            "observations": ["A wide river runs through the northern sector."],
            "objects": ["river", "trees"],
            "uncertainties": []
        }
        validated = SemanticConsistencyValidator.validate_and_normalize(raw)
        assert "none" not in [w.lower() for w in validated["water_features"]]
        assert "water body (observed)" in validated["water_features"]
        assert validated["confidence"] == "high"

    def test_built_up_contradiction_resolution(self):
        raw = {
            "land_cover": ["built-up"],
            "urban_features": ["none"],
            "observations": ["Multiple rectangular rooftops aligned along roads."],
            "objects": [],
            "uncertainties": []
        }
        validated = SemanticConsistencyValidator.validate_and_normalize(raw)
        assert "none" not in [u.lower() for u in validated["urban_features"]]
        assert "buildings" in validated["objects"]
        assert "roads" in validated["objects"]

    def test_dynamic_confidence_low_on_high_uncertainty(self):
        raw = {
            "observations": ["Dark patch visible."],
            "objects": [],
            "uncertainties": ["Low resolution", "Cloud coverage", "Unknown sensor"]
        }
        validated = SemanticConsistencyValidator.validate_and_normalize(raw)
        assert validated["confidence"] == "moderate" or validated["confidence"] == "low"


class TestSceneContextAndControllerIntegration:
    """Validates single source of truth in SceneContext and AgentController."""

    def test_scene_context_set_qwen_visual_context(self):
        ctx = SceneContext()
        vc = {
            "image_type": "optical_satellite",
            "summary": "Dense urban area with multi-story buildings.",
            "objects": ["buildings", "bridges"],
            "uncertainties": ["obscured by cloud shadow"]
        }
        ctx.set_qwen_visual_context(vc)
        assert ctx.scene_understanding["summary"] == "Dense urban area with multi-story buildings."
        assert len(ctx.objects) == 2
        assert ctx.objects[0]["label"] == "buildings"
        assert "obscured by cloud shadow" in ctx.limitations

    def test_agent_controller_visual_context_property(self):
        ctrl = AgentController()
        assert hasattr(ctrl, "qwen_vision")
        assert hasattr(ctrl, "extract_visual_context")

