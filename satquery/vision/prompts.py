"""
Prompt templates and targeted question suites for Visual Scene Understanding.
"""

SCENE_PROMPTS = {
    "system_scene_analyst": (
        "You are an Earth Observation Visual Analyst. Based strictly on the provided visual evidence probes, "
        "synthesize a factual, grounded overview of the remote sensing scene. "
        "Identify the primary land-use type, visible structures, road networks, vegetation presence, and spatial layout. "
        "Never invent features not supported by the visual evidence."
    )
}

OVERVIEW_VQA_QUESTIONS = [
    {"key": "scene_type", "question": "What type of scene or land cover is shown?"},
    {"key": "buildings", "question": "Are buildings or urban structures visible?"},
    {"key": "roads", "question": "Are roads or transportation networks visible?"},
    {"key": "vegetation", "question": "Is vegetation or green cover visible?"},
    {"key": "water", "question": "Is water or a waterbody visible?"}
]
