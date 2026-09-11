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
    "vqa": ToolDefinition(
        name="vqa",
        description="ViLT-B32 Visual Question Answering for scene queries.",
        inputs=["image_1", "question"],
        outputs=["answer", "confidence", "top_predictions"],
        specialist_type="ViLT-B32 (Vision-Language Transformer)",
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
    "geospatial": ToolDefinition(
        name="geospatial",
        description="Rasterio GeoTIFF georeferencing, CRS inspection, bounds, and coordinate transformation.",
        inputs=["image_1"],
        outputs=["crs", "bounds", "resolution", "affine_transform", "has_geospatial_meta"],
        specialist_type="Rasterio Geospatial Engine",
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
    "build_summary": ToolDefinition(
        name="build_summary",
        description="Compiles a factual executive summary from actual specialist results.",
        inputs=["evidence"],
        outputs=["finding", "largest_change", "evidence_source", "geospatial_info", "limitations"],
        specialist_type="Analyst Synthesis Engine",
        is_neural=False
    ),
    "build_report": ToolDefinition(
        name="build_report",
        description="Generates a structured markdown intelligence report from real analysis results.",
        inputs=["result_id"],
        outputs=["markdown_report"],
        specialist_type="Report Generator",
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
                # Handle both ymin,xmin,ymax,xmax and xmin,ymin,xmax,ymax
                h = abs(box[2] - box[0])
                w = abs(box[3] - box[1])
                area = h * w
                clusters.append({
                    "index": idx,
                    "bbox": box,
                    "width": w,
                    "height": h,
                    "area_pixels": area,
                    "rank": 0
                })

        # Sort by area descending
        clusters.sort(key=lambda c: c["area_pixels"], reverse=True)
        total_area = sum(c["area_pixels"] for c in clusters) or 1.0

        for rank_idx, c in enumerate(clusters):
            c["rank"] = rank_idx + 1
            c["relative_percentage"] = round((c["area_pixels"] / total_area) * 100, 2)

        return clusters

    @staticmethod
    def get_cluster(clusters: List[Dict[str, Any]], cluster_index: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific cluster by index."""
        for c in clusters:
            if c["index"] == cluster_index:
                return c
        if 0 <= cluster_index < len(clusters):
            return clusters[cluster_index]
        return None
