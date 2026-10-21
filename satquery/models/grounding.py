"""
Remote Sensing Spatial Text Grounding Engine for SatQuery AI.
Locates queried objects or terrain features from natural language queries, generates
bounding boxes, pixel masks, visual overlays, and geographic coordinates.
"""

# HF cache uses default standard cache or current workspace


from typing import Dict, Any, Union, Optional, List, Tuple
from collections import OrderedDict
import threading
import os
import numpy as np
from PIL import Image
import cv2
import torch
import logging
from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTImageProcessor, CLIPTokenizer
from huggingface_hub import hf_hub_download

logger = logging.getLogger("satquery.grounding")


class TextGroundingEngine:
    """
    Open-Vocabulary Spatial Grounding Engine for Remote Sensing Imagery.
    Extracts deep multi-scale feature representations to localize text-queried targets
    into bounding boxes [ymin, xmin, ymax, xmax], segmentation contours, and visual overlays.
    Optimized with thread-safe LRU caching and torch.inference_mode.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_size = 64
        
        # Load Pre-Trained OWL-ViT for Open Vocabulary Object Detection
        try:
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        except Exception as exc:
            logger.info("Direct OwlViTProcessor loading failed (%s); loading components individually...", exc)
            vocab = hf_hub_download("google/owlvit-base-patch32", "vocab.json")
            merges = hf_hub_download("google/owlvit-base-patch32", "merges.txt")
            tokenizer = CLIPTokenizer(vocab_file=vocab, merges_file=merges)
            image_processor = OwlViTImageProcessor.from_pretrained("google/owlvit-base-patch32")
            self.processor = OwlViTProcessor(image_processor=image_processor, tokenizer=tokenizer)

        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _compute_image_key(img: Union[str, np.ndarray, Image.Image]) -> str:
        """Fast key computation for image inputs."""
        if isinstance(img, str):
            if os.path.isfile(img):
                try:
                    mtime = os.path.getmtime(img)
                    return f"file:{img}:{mtime}"
                except OSError:
                    pass
            return f"path:{img}"
        elif isinstance(img, Image.Image):
            return f"pil:{img.size}:{img.mode}:{id(img)}"
        elif isinstance(img, np.ndarray):
            return f"np:{img.shape}:{img.dtype}:{id(img)}"
        return f"obj:{id(img)}"

    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[Image.Image, np.ndarray]:
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.fromarray(image_input[:, :, :3].astype(np.uint8))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_input)}")
            
        np_img = np.array(pil_img)
        return pil_img, np_img

    def ground_text(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        target_text: str = "built-up structures",
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Locates target regions from the prompt text, generating bounding boxes and overlay visuals.
        Accelerated with thread-safe bounded LRU caching and torch.inference_mode.
        """
        clean_target = target_text.strip()
        img_key = self._compute_image_key(image_input)
        cache_key = f"{img_key}::{clean_target.lower()}::{threshold}"

        with self._cache_lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return dict(self._cache[cache_key])

        pil_img, np_img = self._load_image(image_input)
        orig_w, orig_h = pil_img.size
        
        # Query parsing (OWL-ViT expects lists of lists of strings for batched text)
        texts = [[clean_target]]
        
        # Preprocess and forward
        inputs = self.processor(text=texts, images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        # Target image sizes (height, width) to rescale bounding boxes
        target_sizes = torch.tensor([pil_img.size[::-1]]).to(self.device)
        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        results = self.processor.image_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
        
        # Get the first (and only) image's results
        result = results[0]
        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
        
        bboxes: List[List[int]] = []
        box_scores: List[float] = []
        
        # Binary mask generation
        binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box.tolist()
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            
            # Ensure within bounds
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(orig_w, xmax), min(orig_h, ymax)
            
            # To match the original schema: [ymin, xmin, ymax, xmax]
            bboxes.append([ymin, xmin, ymax, xmax])
            box_scores.append(round(float(score), 4))
            
            # Draw on mask
            cv2.rectangle(binary_mask, (xmin, ymin), (xmax, ymax), 255, -1)
            
        # Generate Visual Grounding Overlay
        overlay_np = self._render_grounding_overlay(np_img, bboxes, box_scores, clean_target)
        
        # Average grounding confidence
        overall_conf = float(np.mean(box_scores)) if box_scores else 0.0
        
        explanation = (
            f"Successfully grounded {len(bboxes)} candidate region{'s' if len(bboxes) != 1 else ''} "
            f"matching target '{clean_target}' (mean confidence: {overall_conf * 100:.1f}%)."
        )
        
        res = {
            "status": "EXECUTED",
            "task": "grounding",
            "target_text": clean_target,
            "explanation": explanation,
            "bboxes": bboxes,
            "box_confidences": box_scores,
            "confidence": overall_conf,
            "confidence_source": "OWL-ViT Open-Vocabulary Object Detection",
            "num_regions": len(bboxes),
            "target_mask": binary_mask,
            "visualization": overlay_np,
            "model_name": "google/owlvit-base-patch32"
        }

        with self._cache_lock:
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)
            self._cache[cache_key] = res

        return res

    def _render_grounding_overlay(
        self,
        img_np: np.ndarray,
        bboxes: List[List[int]],
        scores: List[float],
        label_text: str
    ) -> np.ndarray:
        """Renders vibrant bounding boxes, semi-transparent masks, and labels on satellite image."""
        vis = img_np.copy()
        overlay = vis.copy()
        
        for [ymin, xmin, ymax, xmax], score in zip(bboxes, scores):
            # Highlight region with turquoise mask
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 230, 255), -1)
            # Draw crisp border
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 240, 255), 2)
            
            # Add text badge
            text_str = f"{label_text} ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (xmin, max(0, ymin - th - 8)), (xmin + tw + 6, max(th + 8, ymin)), (0, 180, 220), -1)
            cv2.putText(vis, text_str, (xmin + 3, max(th + 2, ymin - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            
        # Blend overlay (alpha = 0.25)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        return vis
