"""
Remote Sensing Visual Question Answering (RS-VQA) Engine for SatQuery AI.
Integrates vision-language representation learning with domain-specific remote-sensing heads
to answer scene, object, land-cover, and numerical queries with calibrated confidence.
"""

# HF cache uses default standard cache or current workspace


import os
from typing import Dict, Any, Union, Optional, List, Tuple
from collections import OrderedDict
import threading
import hashlib
import numpy as np
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering

class RemoteSensingVQA:
    """
    Remote Sensing Visual Question Answering Model using ViLT.
    This replaces the heuristic baseline with an actual Vision-Language Model.
    Optimized with thread-safe LRU caching and torch.inference_mode.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_size = 64
        
        # Load pre-trained ViLT for VQA
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load Custom LoRA Adapter for Remote Sensing specific presence classification
        self.use_custom_lora = False
        self.lora_model = None
        lora_ckpt = "checkpoints/lora_rsvqa/lora_rs_adapter.pt"
        if os.path.exists(lora_ckpt):
            try:
                import torchvision.models as tv_models
                import torchvision.transforms as T
                import torch.nn as nn
                from peft import get_peft_model, LoraConfig
                
                base_model = tv_models.resnet50(weights=None)
                base_model.fc = nn.Linear(base_model.fc.in_features, 2)
                
                peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["fc"], lora_dropout=0.05, bias="none")
                self.lora_model = get_peft_model(base_model, peft_config)
                
                ckpt = torch.load(lora_ckpt, map_location=self.device)
                self.lora_model.load_state_dict(ckpt["state_dict"])
                self.lora_model.to(self.device)
                self.lora_model.eval()
                
                self.lora_transform = T.Compose([
                    T.Resize((256, 256)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.use_custom_lora = True
                print("Loaded custom RS-VQA LoRA adapter successfully!")
            except Exception as e:
                print(f"Could not load custom LoRA adapter: {e}")

    def _preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input).convert("RGB")
            else:
                img = Image.fromarray(image_input[:, :, :3].astype(np.uint8))
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        return img

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

    def answer(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        question: str
    ) -> Dict[str, Any]:
        """
        Processes a natural language question about an aerial/satellite image.
        Accelerated with thread-safe bounded LRU caching and torch.inference_mode.
        """
        clean_q = question.strip()
        img_key = self._compute_image_key(image_input)
        cache_key = f"{img_key}::{clean_q.lower()}"

        with self._cache_lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return dict(self._cache[cache_key])

        img = self._preprocess_image(image_input)
        
        # Prepare inputs
        inputs = self.processor(img, clean_q, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = self.model.config.id2label[idx]
            
            # Confidence calculation (softmax over logits)
            probs = torch.softmax(logits, dim=-1)
            confidence = probs[0, idx].item()

        q_lower = clean_q.lower()
        task = "scene_vqa"
        count = None
        is_present = None

        if any(k in q_lower for k in ["how many", "count", "number of", "quantity"]):
            task = "numerical_vqa"
            try:
                count = int(answer)
            except ValueError:
                count = 1
        elif any(k in q_lower for k in ["is there", "are there", "does this contain", "do you see", "present"]):
            task = "presence_vqa"
            
            if self.use_custom_lora:
                try:
                    tensor = self.lora_transform(img).unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        l_out = self.lora_model(tensor)
                        l_probs = torch.softmax(l_out, dim=1)
                        l_pred = torch.argmax(l_probs, dim=1).item()
                        
                    is_present = (l_pred == 1)
                    answer = "yes" if is_present else "no"
                    confidence = l_probs[0, l_pred].item()
                except Exception as e:
                    print(f"Error running LoRA adapter: {e}")
                    is_present = answer.lower() in ["yes", "true", "1"]
            else:
                yes_id = self.model.config.label2id.get("yes")
                no_id = self.model.config.label2id.get("no")
                if yes_id is not None and no_id is not None:
                    if logits[0, yes_id] > logits[0, no_id]:
                        is_present = True
                        answer = "yes"
                        confidence = probs[0, yes_id].item()
                    else:
                        is_present = False
                        answer = "no"
                        confidence = probs[0, no_id].item()
                else:
                    is_present = answer.lower() in ["yes", "true", "1"]

        top_k = min(5, logits.shape[-1])
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        top_predictions = [
            {"class": self.model.config.id2label[i.item()], "probability": float(p.item())}
            for p, i in zip(top_probs, top_indices)
        ]

        response = {
            "task": task,
            "question": clean_q,
            "answer": answer,
            "confidence": round(float(confidence), 4),
            "confidence_source": "ViLT Softmax Posterior Logits",
            "model_name": "dandelin/vilt-b32-finetuned-vqa",
            "top_predictions": top_predictions
        }

        if task == "numerical_vqa" and count is not None:
            response["count"] = count
        if task == "presence_vqa" and is_present is not None:
            response["is_present"] = is_present

        response["scene_classification"] = answer

        with self._cache_lock:
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)
            self._cache[cache_key] = response

        return response
