"""
ChangeDetectionModel: Adapter wrapper around ChangeFormerV6 for SatQuery AI.
Preserves existing ChangeFormer model implementation without modifying original files.
"""

import os
from types import SimpleNamespace
from typing import Dict, Any, Union, Tuple, List, Optional
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms.functional as TF

from collections import OrderedDict
import threading
import hashlib

from models.networks import define_G


class ChangeDetectionModel:
    """
    Adapter wrapper for ChangeFormer change detection model.
    Conforms to SatQuery AI modular specialist architecture.
    """

    DEFAULT_CHECKPOINT_DIRS = [
        os.path.join("checkpoints", "ChangeFormer_LEVIR_TEST"),
        os.path.join("checkpoints", "ChangeFormer_LEVIR"),
        os.path.join(
            "checkpoints",
            "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256"
        )
    ]

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_name: str = "best_ckpt.pt",
        img_size: int = 256,
        embed_dim: int = 256,
        net_G: str = "ChangeFormerV6",
        gpu_ids: Optional[List[int]] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.net_G_name = net_G

        if gpu_ids is None:
            gpu_ids = [0] if torch.cuda.is_available() else []
        self.gpu_ids = gpu_ids

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                f"cuda:{gpu_ids[0]}" if (torch.cuda.is_available() and len(gpu_ids) > 0) else "cpu"
            )

        # Locate checkpoint
        self.checkpoint_path = self._resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name
        )

        # Build args object expected by define_G
        self.args = SimpleNamespace(
            n_class=2,
            embed_dim=self.embed_dim,
            net_G=self.net_G_name,
            gpu_ids=self.gpu_ids
        )

        self.net_G = None
        self.loaded_meta = {}
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_size = 32
        self._initialize_model()

    @staticmethod
    def _compute_image_key(img: Union[str, np.ndarray, Image.Image]) -> str:
        """Fast key computation for image inputs without deep hash overhead."""
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

    def _resolve_checkpoint_path(
        self,
        checkpoint_path: Optional[str],
        checkpoint_dir: Optional[str],
        checkpoint_name: str
    ) -> str:
        if checkpoint_path and os.path.isfile(checkpoint_path):
            return checkpoint_path

        if checkpoint_dir:
            full_path = os.path.join(checkpoint_dir, checkpoint_name)
            if os.path.isfile(full_path):
                return full_path

        # Search default locations
        for candidate_dir in self.DEFAULT_CHECKPOINT_DIRS:
            candidate_path = os.path.join(candidate_dir, checkpoint_name)
            if os.path.isfile(candidate_path):
                return candidate_path

        raise FileNotFoundError(
            f"Could not find ChangeFormer checkpoint '{checkpoint_name}' in specified paths "
            f"or default directories: {self.DEFAULT_CHECKPOINT_DIRS}"
        )

    def _initialize_model(self):
        """Instantiates network and loads weights safely."""
        self.net_G = define_G(args=self.args, gpu_ids=self.gpu_ids)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        if "model_G_state_dict" in checkpoint:
            self.net_G.load_state_dict(checkpoint["model_G_state_dict"])
            self.loaded_meta["best_val_acc"] = checkpoint.get("best_val_acc", None)
            self.loaded_meta["best_epoch_id"] = checkpoint.get("best_epoch_id", None)
        else:
            self.net_G.load_state_dict(checkpoint)

        self.net_G.to(self.device)
        self.net_G.eval()

    def _preprocess_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Loads and standardizes input to [1, 3, img_size, img_size] normalized tensor.
        Returns tensor and original (width, height).
        """
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image_input = np.stack([image_input] * 3, axis=-1)
            elif image_input.shape[2] == 4:
                image_input = image_input[:, :, :3]
            pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_size = pil_img.size  # (width, height)

        # Resize to model input size
        resized_img = pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to tensor and normalize to [-1, 1] as expected by ChangeFormer
        tensor_img = TF.to_tensor(resized_img)
        norm_tensor = TF.normalize(tensor_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        norm_tensor = norm_tensor.unsqueeze(0)  # [1, 3, H, W]

        return norm_tensor, orig_size

    def _extract_bounding_boxes(
        self,
        mask: np.ndarray,
        min_area: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Extracts bounding boxes and stats for connected change regions.
        """
        binary = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        bboxes = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx, cy = float(centroids[i][0]), float(centroids[i][1])
                bboxes.append({
                    "bbox": [x, y, x + w, y + h],
                    "area": area,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "width": w,
                    "height": h
                })

        # Sort bboxes by area descending
        bboxes.sort(key=lambda b: b["area"], reverse=True)
        return bboxes

    def _create_overlay(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 50, 50),
        alpha: float = 0.45
    ) -> np.ndarray:
        """
        Creates a visual overlay of detected changes on the base image.
        """
        if isinstance(image, Image.Image):
            base = np.array(image.convert("RGB"))
        else:
            base = image.copy()
            if base.ndim == 2:
                base = np.stack([base] * 3, axis=-1)

        # Resize mask if shapes differ
        if mask.shape[:2] != base.shape[:2]:
            mask_resized = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        change_idx = mask_resized > 0
        overlay = base.copy()
        for c in range(3):
            overlay[change_idx, c] = (
                (1 - alpha) * base[change_idx, c] + alpha * color[c]
            ).astype(np.uint8)

        # Draw subtle contours
        contours, _ = cv2.findContours((mask_resized > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

        return overlay

    def predict(
        self,
        image_t1: Union[str, np.ndarray, Image.Image],
        image_t2: Union[str, np.ndarray, Image.Image],
        resize_to_original: bool = True,
        change_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Performs bi-temporal change detection on image pair T1 and T2.
        Accelerated with thread-safe bounded LRU caching and torch.inference_mode.
        """
        # Cache lookup
        k1 = self._compute_image_key(image_t1)
        k2 = self._compute_image_key(image_t2)
        cache_key = f"{k1}::{k2}::{resize_to_original}::{change_threshold}"

        with self._cache_lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                cached = self._cache[cache_key]
                return dict(cached)

        # Preprocess both images
        tensor_t1, orig_size_t1 = self._preprocess_image(image_t1)
        tensor_t2, orig_size_t2 = self._preprocess_image(image_t2)

        tensor_t1 = tensor_t1.to(self.device)
        tensor_t2 = tensor_t2.to(self.device)

        # Inference with torch.inference_mode
        with torch.inference_mode():
            outputs = self.net_G(tensor_t1, tensor_t2)
            # ChangeFormer outputs multi-scale list or single logits tensor
            logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs

            # Class 0: unchanged, Class 1: changed
            probs = torch.softmax(logits, dim=1)
            change_prob = probs[0, 1, :, :].cpu().numpy()
            pred_class = (change_prob >= change_threshold).astype(np.uint8)

            # Model confidence: mean softmax probability of chosen classes
            max_probs = torch.max(probs, dim=1)[0][0].cpu().numpy()
            mean_confidence = float(np.mean(max_probs))

        # Handle output dimensions
        target_size = orig_size_t1 if resize_to_original else (self.img_size, self.img_size)
        if resize_to_original and (target_size != (self.img_size, self.img_size)):
            change_mask_255 = cv2.resize(
                (pred_class * 255).astype(np.uint8),
                target_size,
                interpolation=cv2.INTER_NEAREST
            )
            prob_map = cv2.resize(
                change_prob,
                target_size,
                interpolation=cv2.INTER_LINEAR
            )
        else:
            change_mask_255 = (pred_class * 255).astype(np.uint8)
            prob_map = change_prob

        # Calculate metrics
        binary_mask = (change_mask_255 > 0).astype(np.uint8)
        changed_pixels = int(np.sum(binary_mask == 1))
        total_pixels = int(binary_mask.size)
        change_percentage = round(float((changed_pixels / total_pixels) * 100.0), 4)

        # Bounding boxes for changed components
        bboxes = self._extract_bounding_boxes(binary_mask)

        # Overlays
        # Load raw images for overlay generation
        raw_t1 = image_t1 if isinstance(image_t1, (np.ndarray, Image.Image)) else Image.open(image_t1)
        raw_t2 = image_t2 if isinstance(image_t2, (np.ndarray, Image.Image)) else Image.open(image_t2)

        overlay_t1 = self._create_overlay(raw_t1, binary_mask)
        overlay_t2 = self._create_overlay(raw_t2, binary_mask)

        # Difference heatmap visualization (Jet colormap on change probabilities)
        diff_heat = (prob_map * 255).astype(np.uint8)
        diff_colored = cv2.applyColorMap(diff_heat, cv2.COLORMAP_JET)
        diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

        result = {
            "change_mask": change_mask_255,
            "change_percentage": change_percentage,
            "confidence": round(mean_confidence, 4),
            "raw_probabilities": prob_map,
            "difference_map": diff_colored,
            "overlay_t1": overlay_t1,
            "overlay_t2": overlay_t2,
            "bboxes": bboxes,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "metadata": {
                "model_name": self.net_G_name,
                "checkpoint": os.path.basename(self.checkpoint_path),
                "device": str(self.device),
                "input_resolution": f"{target_size[0]}x{target_size[1]}",
                "processed_resolution": f"{self.img_size}x{self.img_size}",
                "num_change_clusters": len(bboxes)
            }
        }

        with self._cache_lock:
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)
            self._cache[cache_key] = result

        return result

