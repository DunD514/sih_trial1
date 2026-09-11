"""
Dedicated Text-to-Speech (TTS) Service for SatQuery AI — JARVIS 2.0.
Provides neural speech generation via edge-tts with pyttsx3 fallback,
natural speech normalization for technical metrics, speech queue management,
and audio caching.
"""

import os
import re
import io
import base64
import asyncio
from typing import Dict, Any, Optional, Tuple

# Centralized TTS Configuration
TTS_ENABLED = True
TTS_PROVIDER = "edge-tts"  # Network-dependent neural TTS provider
TTS_VOICE = "en-US-ChristopherNeural"  # Professional aerospace analyst voice
TTS_VOICE_ALT = "en-US-GuyNeural"
TTS_SPEED = 1.0
TTS_PITCH = 1.0
TTS_VOLUME = 0.9

# Number word dictionary for speech normalization
_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"
}
_TENS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"
}


def _int_to_words(n: int) -> str:
    """Convert an integer (0 to 999,999,999) into spoken words."""
    if n < 0:
        return "negative " + _int_to_words(-n)
    if n in _ONES:
        return _ONES[n]
    if n < 100:
        return _TENS[n - (n % 10)] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rem = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rem == 0 else " and " + _int_to_words(rem))
    if n < 1000000:
        thousands = n // 1000
        rem = n % 1000
        return _int_to_words(thousands) + " thousand" + ("" if rem == 0 else " " + _int_to_words(rem))
    millions = n // 1000000
    rem = n % 1000000
    return _int_to_words(millions) + " million" + ("" if rem == 0 else " " + _int_to_words(rem))


def normalize_for_speech(text: str) -> str:
    """
    Normalize technical notations into natural spoken language:
    - 20.66% -> "twenty point six six percent"
    - ChangeFormerV6 -> "ChangeFormer V6"
    - ViLT-B32 -> "ViLT B32"
    - OWL-ViT -> "Owl Vit"
    - 13,540 -> "thirteen thousand five hundred forty"
    - CRS EPSG:4326 -> "CRS E P S G forty-three twenty-six"
    - bbox [10, 20, 30, 40] -> "bounding box"
    """
    if not text:
        return ""

    s = text.strip()

    # 1. Clean markdown and brackets
    s = re.sub(r"[*_`#]", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s)  # markdown links

    # 2. Normalize Model names
    s = re.sub(r"ChangeFormerV6", "ChangeFormer V6", s, flags=re.IGNORECASE)
    s = re.sub(r"ChangeFormer", "ChangeFormer", s, flags=re.IGNORECASE)
    s = re.sub(r"ViLT-B32|ViLT_B32", "Vilt B32", s, flags=re.IGNORECASE)
    s = re.sub(r"OWL-ViT|OWL_ViT", "Owl Vit", s, flags=re.IGNORECASE)
    s = re.sub(r"GeoTIFF", "Geo-Tiff", s, flags=re.IGNORECASE)
    s = re.sub(r"LEVIR-CD", "Levir C D", s, flags=re.IGNORECASE)
    s = re.sub(r"DSIFN", "D S I F N", s, flags=re.IGNORECASE)
    s = re.sub(r"EPSG:?(\d+)", r"E P S G \1", s, flags=re.IGNORECASE)

    # 3. Percentages with decimal: e.g., 20.66% -> "twenty point six six percent"
    def _replace_pct_decimal(m):
        integer_part = int(m.group(1))
        decimals = " ".join(_ONES.get(int(d), d) for d in m.group(2))
        return f"{_int_to_words(integer_part)} point {decimals} percent"

    s = re.sub(r"\b(\d+)\.(\d+)\s*%", _replace_pct_decimal, s)

    # 4. Whole percentages: e.g., 25% -> "twenty-five percent"
    def _replace_pct_int(m):
        val = int(m.group(1))
        return f"{_int_to_words(val)} percent"

    s = re.sub(r"\b(\d+)\s*%", _replace_pct_int, s)

    # 5. Formatted integers with commas: e.g., 13,540 -> "thirteen thousand five hundred forty"
    def _replace_comma_nums(m):
        try:
            num = int(m.group(0).replace(",", ""))
            return _int_to_words(num)
        except Exception:
            return m.group(0)

    s = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", _replace_comma_nums, s)

    # 6. Isolated decimals: e.g., 0.98 -> "zero point nine eight"
    def _replace_decimal(m):
        integer_part = int(m.group(1))
        decimals = " ".join(_ONES.get(int(d), d) for d in m.group(2))
        return f"{_int_to_words(integer_part)} point {decimals}"

    s = re.sub(r"\b(\d+)\.(\d{1,4})\b", _replace_decimal, s)

    # 7. Coordinates / Bounding boxes in text
    s = re.sub(r"\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]", "the selected bounding box", s)

    # 8. Clean excess whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def prepare_speech_text(text: str, max_words: int = 40) -> str:
    """
    Format and prune text for conversational voice playback (1-2 crisp sentences).
    """
    normalized = normalize_for_speech(text)
    words = normalized.split()
    if len(words) > max_words:
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        truncated = sentences[0]
        if len(sentences) > 1 and len(truncated.split()) < 20:
            truncated += " " + sentences[1]
        return truncated
    return normalized


class TTSService:
    """
    Master TTS Service managing Neural Audio generation, local fallback,
    and voice status tracking.
    """

    def __init__(self, default_voice: str = TTS_VOICE):
        self.default_voice = default_voice
        self._cache: Dict[str, bytes] = {}
        self._pyttsx3_engine = None

    async def get_status(self) -> Dict[str, Any]:
        """Return runtime status of TTS engines."""
        return {
            "tts_enabled": TTS_ENABLED,
            "provider": TTS_PROVIDER,
            "voice": self.default_voice,
            "voice_alt": TTS_VOICE_ALT,
            "status": "ready" if TTS_ENABLED else "disabled",
            "is_neural": True,
            "note": "edge-tts is network-dependent neural TTS; browser SpeechSynthesis provides local fallback."
        }

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: float = TTS_SPEED
    ) -> Tuple[Optional[bytes], str, str]:
        """
        Synthesize text into MP3 audio bytes.
        Returns (audio_bytes, provider_used, voice_used).
        """
        if not TTS_ENABLED or not text:
            return None, "disabled", ""

        normalized = prepare_speech_text(text)
        selected_voice = voice or self.default_voice
        cache_key = f"{selected_voice}:{normalized}"

        if cache_key in self._cache:
            return self._cache[cache_key], TTS_PROVIDER, selected_voice

        # 1. Primary: edge-tts
        try:
            import edge_tts
            communicate = edge_tts.Communicate(normalized, selected_voice)
            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            audio_bytes = audio_buffer.getvalue()
            if audio_bytes:
                # Keep cache small (max 50 items)
                if len(self._cache) > 50:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = audio_bytes
                return audio_bytes, "edge-tts", selected_voice
        except Exception as exc:
            print(f"[TTSService] edge-tts error ({exc}), attempting pyttsx3 fallback...")

        # 2. Secondary: pyttsx3 fallback (offline WAV/MP3)
        try:
            import pyttsx3
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            def _run_pyttsx3():
                engine = pyttsx3.init()
                engine.setProperty('rate', int(175 * rate))
                engine.save_to_file(normalized, tmp_path)
                engine.runAndWait()

            await asyncio.to_thread(_run_pyttsx3)

            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with open(tmp_path, "rb") as f:
                    wav_bytes = f.read()
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return wav_bytes, "pyttsx3", "local_sapi5"
        except Exception as exc2:
            print(f"[TTSService] pyttsx3 fallback also failed: {exc2}")

        return None, "browser_fallback", ""

    async def synthesize_base64(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate base64 encoded audio for instant frontend playback."""
        audio_bytes, provider, voice_used = await self.synthesize(text, voice)
        if audio_bytes:
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            mime_type = "audio/wav" if provider == "pyttsx3" else "audio/mp3"
            return {
                "success": True,
                "audio_base64": f"data:{mime_type};base64,{b64_audio}",
                "provider": provider,
                "voice": voice_used,
                "normalized_text": prepare_speech_text(text)
            }
        return {
            "success": False,
            "audio_base64": None,
            "provider": "browser_fallback",
            "voice": "browser",
            "normalized_text": prepare_speech_text(text)
        }


# Global singleton instance
tts_service = TTSService()
