"""
Lightweight unit tests for JARVIS UI Action Execution and Critical Heatmap Transition.
"""

import sys
import os

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.agent.context import SceneContext
from satquery.agent.planner import NaturalLanguagePlanner
from satquery.agent.tools import ALLOWED_UI_ACTIONS, UIActionBuilder


class MockFrontend:
    """Mock frontend state receiver."""
    def __init__(self):
        self.active_tab = "overview"
        self.active_view = "overview"
        self.active_overlay = None
        self.selected_cluster = None
        self.evidence_open = False
        self.trace_open = False
        self.zoom_target = None

    def handle_action(self, act: dict):
        name = act.get("action")
        if name == "navigate_tab":
            tab = act.get("tab")
            self.active_tab = tab
            if tab == "diff-map":
                self.active_view = "heatmap"
                self.active_overlay = "heatmap"
            elif tab == "change-mask":
                self.active_view = "change_mask"
                self.active_overlay = "change_mask"
            elif tab == "interactive-overlay":
                self.active_view = "interactive_overlay"
        elif name in ("show_heatmap", "show_diff_map"):
            self.active_tab = "diff-map"
            self.active_view = "heatmap"
            self.active_overlay = "heatmap"
        elif name == "show_change_mask":
            self.active_tab = "change-mask"
            self.active_view = "change_mask"
            self.active_overlay = "change_mask"
        elif name == "show_overlay":
            ov = act.get("overlay", "interactive-overlay")
            self.active_overlay = ov
            if ov in ("heatmap", "diff-map"):
                self.active_tab = "diff-map"
                self.active_view = "heatmap"
            else:
                self.active_tab = "interactive-overlay"
                self.active_view = "overlay"
        elif name in ("select_cluster", "highlight_cluster"):
            self.selected_cluster = act.get("index", 0)
        elif name == "zoom_to_cluster":
            self.zoom_target = act.get("index", 0)
        elif name in ("open_evidence", "show_evidence"):
            self.evidence_open = True
        elif name in ("open_trace", "show_trace"):
            self.trace_open = True


def test_controlled_ui_registry():
    # Valid allowed UI actions
    for act in ["navigate_tab", "show_overlay", "show_heatmap", "show_change_mask", "select_cluster", "zoom_to_cluster", "open_trace", "open_evidence"]:
        assert act in ALLOWED_UI_ACTIONS

    # Strict refusal of arbitrary execution
    forbidden = ["execute_python", "run_shell", "eval_js", "fs_write", "exec_cmd", "open_url", "os_click"]
    for bad_act in forbidden:
        assert bad_act not in ALLOWED_UI_ACTIONS


def test_critical_heatmap_transition():
    """
    CRITICAL TEST:
    User: 'Show me the heatmap.'
    Verify:
    1. Action produced is show_heatmap / navigate_tab:diff-map / show_overlay:heatmap
    2. Frontend active_view changes to 'heatmap' and active_overlay == 'heatmap'
    """
    ctx = SceneContext()
    ctx.has_image_1 = True
    ctx.has_image_2 = True
    ctx.change_result = {"change_percentage": 20.66, "clusters": [{"id": "region_1", "index": 0}]}

    plan = NaturalLanguagePlanner.classify_and_plan("Show me the heatmap.", ctx)
    assert plan["intent"] in ("UI_COMMAND", "UI_SHOW_HEATMAP")

    ui_actions = UIActionBuilder.build_actions(plan, specialist_result=ctx.change_result)
    assert any(a.get("action") in ("show_heatmap", "show_diff_map") or a.get("tab") == "diff-map" for a in ui_actions)

    fe = MockFrontend()
    for act in ui_actions:
        fe.handle_action(act)

    assert fe.active_view == "heatmap", f"Expected active_view 'heatmap', got '{fe.active_view}'"
    assert fe.active_overlay == "heatmap"
    assert fe.active_tab == "diff-map"


def test_sequential_ui_commands():
    ctx = SceneContext()
    ctx.change_result = {"change_percentage": 20.66, "clusters": [{"id": "region_1", "index": 0}, {"id": "region_2", "index": 1}]}
    fe = MockFrontend()

    # 1. "Show me the heatmap."
    p1 = NaturalLanguagePlanner.classify_and_plan("Show me the heatmap.", ctx)
    for act in UIActionBuilder.build_actions(p1, ctx.change_result):
        fe.handle_action(act)
    assert fe.active_view == "heatmap"

    # 2. "Show the change mask."
    p2 = NaturalLanguagePlanner.classify_and_plan("Show the change mask.", ctx)
    for act in UIActionBuilder.build_actions(p2, ctx.change_result):
        fe.handle_action(act)
    assert fe.active_view == "change_mask"

    # 3. "Show the biggest change."
    p3 = NaturalLanguagePlanner.classify_and_plan("Show the biggest change.", ctx)
    for act in UIActionBuilder.build_actions(p3, ctx.change_result):
        fe.handle_action(act)
    assert fe.selected_cluster == 0

    # 4. "Zoom into it."
    ctx.set_active_cluster(0)
    p4 = NaturalLanguagePlanner.classify_and_plan("Zoom into it.", ctx)
    for act in UIActionBuilder.build_actions(p4, ctx.change_result):
        fe.handle_action(act)
    assert fe.zoom_target == 0

    # 5. "Open the evidence."
    p5 = NaturalLanguagePlanner.classify_and_plan("Open the evidence.", ctx)
    for act in UIActionBuilder.build_actions(p5, ctx.change_result):
        fe.handle_action(act)
    assert fe.evidence_open is True


if __name__ == "__main__":
    test_controlled_ui_registry()
    test_critical_heatmap_transition()
    test_sequential_ui_commands()
    print("[PASS] test_jarvis_actions.py passed all tests successfully.")
