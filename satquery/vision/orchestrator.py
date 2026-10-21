"""
Analysis Orchestrator for SatQuery AI.
Coordinates multimodal satellite image inputs, Qwen2-VL visual context extraction,
specialist remote-sensing analysis engines (ChangeFormer, ViLT, OWL-ViT, Optical/SAR),
and injects structured visual context into the existing Ollama reasoning pipeline.
"""

import os
import re
import json
import logging
import hashlib
import time
from typing import Dict, Any, Optional, Union, List
from PIL import Image

from .qwen_service import qwen_vision_service, Qwen2VLService, EMPTY_VISUAL_CONTEXT
from ..services.ollama import ollama_service, OllamaService

logger = logging.getLogger("satquery.vision.orchestrator")


# ---------------------------------------------------------------------------
# Synthesis response cache — avoids re-running Ollama for repeated queries
# on the same image / intent combination.
# ---------------------------------------------------------------------------
_SYNTHESIS_CACHE: Dict[str, Dict[str, Any]] = {}
_SYNTHESIS_CACHE_TTL: float = 300.0  # 5 minutes
_SYNTHESIS_CACHE_MAX: int = 64


def _synthesis_cache_key(fused_prompt: str, query_intent: str, visual_context: Optional[Dict[str, Any]] = None) -> str:
    """Produces a stable hash key for a synthesis request.
    For generic scene descriptions, keys by visual context content so alternate question
    phrasings on the same image share the cached description."""
    is_generic = query_intent in (
        "GENERIC_IMAGE_DESCRIPTION", "SCENE_UNDERSTANDING",
        "IMAGE_CAPTIONING", "SINGLE_IMAGE_OVERVIEW", "SCENE_DESCRIPTION"
    )
    if is_generic and visual_context:
        summary = visual_context.get("summary", "").strip()
        objects = ",".join(visual_context.get("objects", []))
        raw = f"scene_desc::{summary}::{objects}"
    else:
        raw = f"{query_intent}::{fused_prompt[:1024]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _synthesis_cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _SYNTHESIS_CACHE.get(key)
    if entry and (time.time() - entry["_ts"]) < _SYNTHESIS_CACHE_TTL:
        return entry["value"]
    return None


def _synthesis_cache_put(key: str, value: Dict[str, Any]) -> None:
    global _SYNTHESIS_CACHE
    if len(_SYNTHESIS_CACHE) >= _SYNTHESIS_CACHE_MAX:
        # Evict oldest entry
        oldest = min(_SYNTHESIS_CACHE, key=lambda k: _SYNTHESIS_CACHE[k]["_ts"])
        _SYNTHESIS_CACHE.pop(oldest, None)
    _SYNTHESIS_CACHE[key] = {"_ts": time.time(), "value": value}


class AnalysisOrchestrator:
    """
    Central orchestration layer coordinating:
    1. Image validation & preprocessing
    2. Qwen2-VL visual context extraction
    3. Specialist remote-sensing algorithms (ChangeFormer, ViLT, OWL-ViT, Optical/SAR)
    4. Intent-aware prompt construction, Ollama reasoning, and Hard Response Relevance Guarding.
    """

    def __init__(
        self,
        vision_service: Optional[Qwen2VLService] = None,
        ollama: Optional[OllamaService] = None
    ):
        self.vision_service = vision_service or qwen_vision_service
        self.ollama = ollama or ollama_service

    def extract_visual_context(
        self,
        image: Union[str, Image.Image, bytes],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extracts structured visual context via Qwen2-VL.
        Gracefully handles errors, never crashing caller if vision model is unavailable.
        """
        try:
            return self.vision_service.analyze_image(image, metadata=metadata)
        except Exception as e:
            logger.error("Visual context extraction failed in orchestrator: %s", e)
            return {
                "status": "partial",
                "visual_context": None,
                "vision_error": str(e)
            }

    @staticmethod
    def generate_deterministic_fallback(visual_context: Optional[Dict[str, Any]]) -> str:
        """
        Constructs a concise, natural description directly from Qwen2-VL visual context fields.
        Used as a deterministic guarantee when generic visual queries require direct scene description.
        """
        if not visual_context:
            return "The satellite image is available for visual inspection and remote sensing analysis."

        parts = []
        summary = visual_context.get("summary", "").strip()
        if summary and not summary.startswith("{"):
            if not any(k in summary.lower() for k in ["scene", "satellite", "features"]):
                parts.append(f"Satellite scene overview: {summary}")
            else:
                parts.append(summary)

        land_cover = [x for x in visual_context.get("land_cover", []) if str(x).lower() != "none"]
        land_use = [x for x in visual_context.get("land_use", []) if str(x).lower() != "none"]
        if land_cover or land_use:
            lc_str = ", ".join(land_cover) if land_cover else ""
            lu_str = ", ".join(land_use) if land_use else ""
            if lc_str and lu_str:
                parts.append(f"The scene exhibits {lc_str} land cover, with indications of {lu_str} land use.")
            elif lc_str:
                parts.append(f"Dominant land cover includes {lc_str}.")
            elif lu_str:
                parts.append(f"Identified land use patterns include {lu_str}.")

        objs = [x for x in visual_context.get("objects", []) if str(x).lower() != "none"]
        infra = [x for x in visual_context.get("infrastructure", []) if str(x).lower() != "none"]
        all_structs = list(dict.fromkeys(objs + infra))
        if all_structs:
            parts.append(f"Visible structural elements and infrastructure include {', '.join(all_structs)}.")

        terrain = [x for x in visual_context.get("terrain", []) if str(x).lower() != "none"]
        if terrain:
            parts.append(f"The local topography appears {', '.join(terrain)}.")

        obs = [x for x in visual_context.get("observations", []) if str(x).lower() != "none" and not str(x).startswith("{")]
        if obs:
            parts.append("Key observations: " + "; ".join(obs) + ".")

        return " ".join(parts) if parts else "The imagery displays a structured remote sensing scene with observable land features."

    @staticmethod
    def format_visual_context_block(visual_context: Optional[Dict[str, Any]]) -> str:
        """
        Formats structured visual context into the standard prompt block for Ollama:
        === VISUAL CONTEXT FROM QWEN2-VL ===
        ...
        === END VISUAL CONTEXT ===
        """
        if not visual_context:
            return (
                "=== PRIMARY VISUAL EVIDENCE ===\n"
                "QWEN2-VL VISUAL CONTEXT\n"
                "Visual context not available for this execution.\n"
                "=== END PRIMARY VISUAL EVIDENCE ==="
            )

        summary = visual_context.get("summary", "")
        img_type = visual_context.get("image_type", "optical_satellite")
        objects = ", ".join(visual_context.get("objects", [])) or "None listed"
        land_cover = ", ".join(visual_context.get("land_cover", [])) or "None listed"
        land_use = ", ".join(visual_context.get("land_use", [])) or "None listed"
        infrastructure = ", ".join(visual_context.get("infrastructure", [])) or "None listed"
        terrain = ", ".join(visual_context.get("terrain", [])) or "None listed"
        vegetation = ", ".join(visual_context.get("vegetation", [])) or "None listed"
        water = ", ".join(visual_context.get("water_features", [])) or "None listed"
        agriculture = ", ".join(visual_context.get("agriculture", [])) or "None listed"
        urban = ", ".join(visual_context.get("urban_features", [])) or "None listed"
        industrial = ", ".join(visual_context.get("industrial_features", [])) or "None listed"
        spatial_rel = ", ".join(visual_context.get("spatial_relationships", [])) or "None listed"
        conditions = ", ".join(visual_context.get("notable_conditions", [])) or "None listed"
        observations = "; ".join(visual_context.get("observations", [])) or "None listed"
        inferences = "; ".join(visual_context.get("inferences", [])) or "None listed"
        uncertainties = "; ".join(visual_context.get("uncertainties", [])) or "None listed"

        block = (
            "=== PRIMARY VISUAL EVIDENCE ===\n"
            "=== VISUAL CONTEXT FROM QWEN2-VL ===\n"
            "QWEN2-VL VISUAL CONTEXT\n"
            f"Image Type: {img_type}\n"
            f"Visual Summary: {summary}\n"
            f"Observed Objects: {objects}\n"
            f"Land Cover: {land_cover}\n"
            f"Land Use: {land_use}\n"
            f"Infrastructure: {infrastructure}\n"
            f"Terrain & Topography: {terrain}\n"
            f"Vegetation: {vegetation}\n"
            f"Water Features: {water}\n"
            f"Agriculture Features: {agriculture}\n"
            f"Urban Layout: {urban}\n"
            f"Industrial Features: {industrial}\n"
            f"Spatial Relationships: {spatial_rel}\n"
            f"Notable Visual Conditions: {conditions}\n"
            f"Direct Observations: {observations}\n"
            f"Visual Inferences: {inferences}\n"
            f"Confidence: {visual_context.get('confidence', 'moderate')}\n"
            f"Uncertainties / Limitations: {uncertainties}\n"
            "=== END VISUAL CONTEXT ===\n"
            "=== END PRIMARY VISUAL EVIDENCE ==="
        )
        return block

    def compact_fuse_context_for_ollama(
        self,
        base_prompt: str,
        visual_context: Optional[Dict[str, Any]],
        query_intent: str = "GENERIC_IMAGE_DESCRIPTION",
    ) -> str:
        """
        Compact prompt builder for generic visual queries.
        Produces a much smaller prompt (~60% fewer tokens) that keeps only
        the most informative fields from the Qwen2-VL visual context.
        Used when intent is GENERIC_IMAGE_DESCRIPTION / SCENE_UNDERSTANDING.
        """
        if not visual_context:
            return (
                f"QUERY: {base_prompt}\n\n"
                "No visual context available. Describe the satellite scene briefly."
            )

        # Pick the highest-signal fields only
        summary = visual_context.get("summary", "").strip()
        objects = ", ".join(visual_context.get("objects", [])[:6]) or "none"
        land_cover = ", ".join(visual_context.get("land_cover", [])[:4]) or "none"
        land_use = ", ".join(visual_context.get("land_use", [])[:4]) or "none"
        infra = ", ".join(visual_context.get("infrastructure", [])[:4]) or "none"
        vegetation = ", ".join(visual_context.get("vegetation", [])[:3]) or "none"
        terrain = ", ".join(visual_context.get("terrain", [])[:2]) or "none"
        obs = "; ".join(visual_context.get("observations", [])[:3]) or "none"
        inferences = "; ".join(visual_context.get("inferences", [])[:2]) or "none"

        return (
            f"QUERY: {base_prompt}\n\n"
            f"INTENT: {query_intent}\n\n"
            "VISUAL EVIDENCE (Qwen2-VL structured extraction):\n"
            f"  Summary: {summary}\n"
            f"  Objects: {objects}\n"
            f"  Land Cover: {land_cover}\n"
            f"  Land Use: {land_use}\n"
            f"  Infrastructure: {infra}\n"
            f"  Vegetation: {vegetation}\n"
            f"  Terrain: {terrain}\n"
            f"  Key Observations: {obs}\n"
            f"  Inferences: {inferences}\n\n"
            "RULES: Answer the query concisely as a remote sensing analyst. "
            "Focus only on what is visually observed. "
            "Do NOT mention ChangeFormer, change percentages, or model names. "
            "Do NOT hallucinate GPS coordinates, sensor bands, or dates. "
            "Keep the answer under 120 words."
        )

    def fuse_context_for_ollama(
        self,
        base_prompt: str,
        visual_context: Optional[Dict[str, Any]],
        image_metadata: Optional[Dict[str, Any]] = None,
        existing_analysis_data: Optional[Dict[str, Any]] = None,
        query_intent: str = "GENERIC_IMAGE_DESCRIPTION",
        relevant_specialists: Optional[List[str]] = None
    ) -> str:
        """
        Combines user query, intent, Qwen primary visual context, image metadata,
        and query-conditional specialist evidence into an enriched prompt for Ollama.
        """
        sections = []

        # 1. User Query
        sections.append(f"=== USER QUERY ===\n{base_prompt}\n=== END USER QUERY ===")

        # 2. Query Intent
        is_change_in_prompt = any(w in base_prompt.lower() for w in ["change", "different", "compare"])
        effective_intent = query_intent
        if query_intent == "GENERIC_IMAGE_DESCRIPTION" and is_change_in_prompt:
            effective_intent = "CHANGE_DETECTION"

        sections.append(f"=== QUERY INTENT ===\n{effective_intent}\n=== END QUERY INTENT ===")

        # 3. Primary Visual Context from Qwen2-VL
        v_block = self.format_visual_context_block(visual_context)
        sections.append(v_block)

        # 4. Query-Conditional Specialist Evidence
        is_change_query = effective_intent in ("CHANGE_DETECTION", "CHANGE_ANALYSIS", "IMAGE_COMPARISON", "COMPLEX_ANALYSIS")
        has_relevant_specialists = bool(relevant_specialists) or is_change_query

        if has_relevant_specialists and existing_analysis_data:
            analysis_str = (
                "=== SPECIALIST EVIDENCE ===\n"
                "=== REMOTE SENSING SPECIALIST EVIDENCE ===\n"
                f"{json.dumps(existing_analysis_data, indent=2, default=str)}\n"
                "=== END SPECIALIST EVIDENCE ==="
            )
            sections.append(analysis_str)
        else:
            sections.append(
                "=== SPECIALIST EVIDENCE ===\n"
                "No specialist result is required for this query.\n"
                "=== END SPECIALIST EVIDENCE ==="
            )

        # 5. Image Metadata if available
        if image_metadata:
            meta_str = (
                "=== IMAGE METADATA ===\n"
                f"{json.dumps(image_metadata, indent=2)}\n"
                "=== END IMAGE METADATA ==="
            )
            sections.append(meta_str)

        # 6. Synthesis Rules
        rules_block = (
            "SYNTHESIS RULES:\n"
            "1. Answer the user's actual question directly.\n"
            "2. For GENERIC_IMAGE_DESCRIPTION or SCENE_UNDERSTANDING, prioritize Qwen2-VL visual context.\n"
            "3. Do not introduce change-detection statistics or percentages unless the user explicitly asks about change.\n"
            "4. Do not mention ChangeFormer or change clusters unless change detection is relevant.\n"
            "5. Do not mention internal model names unless explicitly requested.\n"
            "6. Do not invent unsupported details.\n"
            "7. Distinguish observations from inferences internally.\n"
            "8. Give a concise, natural-language remote sensing analyst answer.\n"
            "9. Never answer a generic image-description question with a change-detection result merely because ChangeFormer produced a result."
        )
        sections.append(rules_block)

        return "\n\n".join(sections)

    def check_and_filter_response_relevance(
        self,
        response_text: str,
        query_intent: str,
        visual_context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Hard Response Relevance Guard:
        Ensures generic image description queries are never answered with ChangeFormer statistics.
        """
        if not response_text:
            return self.generate_deterministic_fallback(visual_context)

        cleaned = response_text.strip()
        # Strip thinking tag blocks if present
        cleaned_no_think = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
        text_to_check = (cleaned_no_think or cleaned).lower()

        # Check for forbidden ChangeFormer patterns in generic visual queries
        if query_intent in ("GENERIC_IMAGE_DESCRIPTION", "SCENE_UNDERSTANDING", "IMAGE_CAPTIONING", "SINGLE_IMAGE_OVERVIEW", "SCENE_DESCRIPTION"):
            forbidden_prefixes = [
                "changeformer detected",
                "the change map shows",
                "the difference probability",
                "spatial change was detected",
                "detected spatial change",
                "there are",  # e.g. "there are 1 change clusters"
                "change detection shows"
            ]
            
            is_irrelevant = any(text_to_check.startswith(prefix) for prefix in forbidden_prefixes) or (
                "changeformer" in text_to_check and "spatial change" in text_to_check
            ) or (
                "change cluster" in text_to_check and "%" in text_to_check
            )

            if is_irrelevant:
                logger.warning("Hard Response Relevance Guard triggered: ChangeFormer output detected for generic visual query (%s). Replacing with Qwen fallback.", query_intent)
                return self.generate_deterministic_fallback(visual_context)

        # Scrub qwen3 inline thinking that leaks without <think> tags.
        # When /no_think is ignored, qwen3 starts with "Okay, the user is asking..."
        # or "Let me check..." style monologue before the actual answer.
        # Heuristic: if the response starts with a reasoning preamble, strip up to the
        # first sentence that looks like an actual substantive statement.
        _INLINE_THINK_PREFIXES = (
            "okay,", "alright,", "let me", "i need to", "the user is asking",
            "looking at", "first,", "so,", "well,", "now,", "i'll",
            "the summary mentions", "the visual evidence shows",
        )
        candidate = cleaned_no_think or cleaned
        candidate_lower = candidate.lower()

        # Check for chain-of-thought monologue leaking anywhere in candidate
        _THINKING_PHRASES = (
            "i need to condense",
            "condense this into",
            "focus on what's visually",
            "avoid any inferred",
            "the summary mentions",
            "the prompt mentions",
            "let me condense",
            "let me summarize",
            "i should mention",
            "my instructions are",
            "as per the instructions",
        )
        if any(tp in candidate_lower for tp in _THINKING_PHRASES):
            # Check if there is a substantive final sentence after the thinking
            answer_match = re.search(r"(?:[.\n]|^)\s*(The (?:image|scene|satellite|aerial)[\s\S]+)$", candidate, re.IGNORECASE)
            if answer_match:
                substantive = answer_match.group(1).strip()
                if (
                    len(substantive) > 40
                    and not any(tp in substantive.lower() for tp in _THINKING_PHRASES)
                    and not substantive.lower().endswith("wait")
                ):
                    candidate = substantive
                    candidate_lower = candidate.lower()
                else:
                    logger.warning("[ResponseGuard] Thinking monologue without clean answer detected. Using fallback.")
                    return self.generate_deterministic_fallback(visual_context)
            else:
                logger.warning("[ResponseGuard] Thinking monologue without clean answer detected. Using fallback.")
                return self.generate_deterministic_fallback(visual_context)

        # Scrub trailing thought fragments like 'Wait' or 'Wait.'
        if candidate_lower.endswith(" wait") or candidate_lower.endswith(" wait.") or candidate_lower.endswith(" wait..."):
            candidate = re.sub(r"(?i)\s+wait[.?!]*$", "", candidate).strip()
            candidate_lower = candidate.lower()

        if any(candidate_lower.startswith(p) for p in _INLINE_THINK_PREFIXES):
            # Find the first paragraph break or a sentence that starts with "The image"
            # or "This image" — the real answer usually begins there.
            answer_start_patterns = [
                r"\n\n",                       # paragraph break
                r"(?<=[.!?])\s+The (?:image|scene|satellite|aerial)",
                r"(?<=[.!?])\s+This (?:image|scene|area)",
                r"(?<=[.!?])\s+Based on",
            ]
            extracted = None
            for pattern in answer_start_patterns:
                m = re.search(pattern, candidate)
                if m:
                    extracted = candidate[m.start():].strip()
                    # Must be a substantive answer, not another thinking fragment
                    if len(extracted) > 80 and not any(
                        extracted.lower().startswith(p) for p in _INLINE_THINK_PREFIXES
                    ):
                        break
                    extracted = None

            if extracted:
                logger.info("[ResponseGuard] Stripped %d chars of qwen3 inline thinking preamble.",
                            len(candidate) - len(extracted))
                candidate = extracted
            else:
                # Full response is thinking; fall back to Qwen2-VL deterministic description
                logger.warning("[ResponseGuard] Full response appears to be qwen3 thinking without answer. Using fallback.")
                return self.generate_deterministic_fallback(visual_context)

        return candidate

    async def execute_reasoning_with_ollama(
        self,
        fused_prompt: str,
        system_instruction: Optional[str] = None,
        query_intent: str = "GENERIC_IMAGE_DESCRIPTION",
        visual_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invokes the existing Ollama model (qwen3:8b) for final synthesis and explanation,
        with Hard Response Relevance Guarding and synthesis result caching.
        """
        # --- Synthesis cache lookup ---
        cache_key = _synthesis_cache_key(fused_prompt, query_intent, visual_context)
        cached = _synthesis_cache_get(cache_key)
        if cached:
            logger.info("[Orchestrator] Synthesis cache HIT for intent=%s", query_intent)
            return cached

        messages = []
        eff_sys = system_instruction or "You are an Earth observation satellite image analyst. Provide a direct, factual 1-2 sentence response describing the satellite scene. Do NOT include thinking or conversational preamble."
        messages.append({"role": "system", "content": eff_sys})
        messages.append({"role": "user", "content": fused_prompt})

        # Use token budget adequate for clean 2-sentence responses
        is_generic = query_intent in (
            "GENERIC_IMAGE_DESCRIPTION", "SCENE_UNDERSTANDING",
            "IMAGE_CAPTIONING", "SINGLE_IMAGE_OVERVIEW", "SCENE_DESCRIPTION"
        )
        num_predict = 250 if is_generic else 200

        try:
            t0 = time.time()
            response_text = await self.ollama.chat(
                messages=messages,
                num_predict=num_predict
            )
            elapsed = time.time() - t0
            logger.info("[Orchestrator] Ollama synthesis: %.2fs (%d tokens budget, intent=%s)",
                        elapsed, num_predict, query_intent)

            guarded_response = self.check_and_filter_response_relevance(
                response_text=response_text,
                query_intent=query_intent,
                visual_context=visual_context
            )
            result = {
                "status": "success",
                "analysis": guarded_response,
                "model": self.ollama.model,
                "query_intent": query_intent
            }
            _synthesis_cache_put(cache_key, result)
            return result
        except Exception as exc:
            logger.error("Ollama reasoning call failed: %s", exc)
            fallback_ans = self.generate_deterministic_fallback(visual_context)
            return {
                "status": "partial",
                "analysis": fallback_ans,
                "error": str(exc),
                "model": self.ollama.model,
                "query_intent": query_intent
            }


# Global singleton instance
orchestrator = AnalysisOrchestrator()

