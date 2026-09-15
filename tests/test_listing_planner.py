"""app/tools/listing_planner.py::plan_listing_shots -- one upfront vision
call that decides which N distinct listing shots to produce. Well-formed
JSON is used as-is (trimmed/padded to exactly output_count); malformed JSON
falls back to DEFAULT_SHOT_TYPES entirely."""

import json
from unittest.mock import MagicMock

from app.tools import listing_planner


def _tool(ai_steps=None):
    return MagicMock(ai_steps=ai_steps if ai_steps is not None else {})


SHOT_PLANNER_STEP = {"model_id": "openrouter/router/vision", "model": "google/gemini-2.5-flash"}


# --------------------------------------------------------------------------- #
# well-formed JSON
# --------------------------------------------------------------------------- #

def test_well_formed_json_is_used_as_is(monkeypatch):
    shots_json = json.dumps([
        {"shot_type": "hero", "prompt": "Hero shot, studio lighting"},
        {"shot_type": "side", "prompt": "Side angle, studio lighting"},
    ])
    call_fal_sync = MagicMock(return_value={"output": shots_json})
    monkeypatch.setattr(listing_planner, "call_fal_sync", call_fal_sync)

    shots = listing_planner.plan_listing_shots(
        _tool(ai_steps={"shot_planner": SHOT_PLANNER_STEP}),
        ["https://fal.test/product.png"], None, output_count=2,
    )

    assert shots == [
        {"shot_type": "hero", "prompt": "Hero shot, studio lighting"},
        {"shot_type": "side", "prompt": "Side angle, studio lighting"},
    ]
    call_fal_sync.assert_called_once_with(
        model_id="openrouter/router/vision",
        input_params={
            "image_urls": ["https://fal.test/product.png"],
            "prompt": "Plan exactly 2 distinct listing shots for this product.",
            "system_prompt": listing_planner._DEFAULT_SYSTEM_PROMPT,
            "model": "google/gemini-2.5-flash",
        },
    )


def test_user_prompt_is_appended_as_a_prioritized_hint(monkeypatch):
    call_fal_sync = MagicMock(return_value={"output": json.dumps([{"shot_type": "hero", "prompt": "x"}])})
    monkeypatch.setattr(listing_planner, "call_fal_sync", call_fal_sync)

    listing_planner.plan_listing_shots(
        _tool(ai_steps={"shot_planner": SHOT_PLANNER_STEP}),
        ["https://fal.test/product.png"], "add a beach background", output_count=1,
    )

    sent_prompt = call_fal_sync.call_args.kwargs["input_params"]["prompt"]
    assert "The user specifically wants: add a beach background" in sent_prompt
    assert "Prioritize this above default shot choices" in sent_prompt


def test_json_wrapped_in_markdown_code_fence_is_still_parsed(monkeypatch):
    fenced = "```json\n" + json.dumps([{"shot_type": "hero", "prompt": "x"}]) + "\n```"
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": fenced}))

    shots = listing_planner.plan_listing_shots(_tool(), [], None, output_count=1)

    assert shots == [{"shot_type": "hero", "prompt": "x"}]


# --------------------------------------------------------------------------- #
# malformed JSON -- falls back to DEFAULT_SHOT_TYPES entirely
# --------------------------------------------------------------------------- #

def test_malformed_json_falls_back_to_default_shots(monkeypatch):
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": "not json at all"}))

    shots = listing_planner.plan_listing_shots(_tool(), [], None, output_count=3)

    assert [s["shot_type"] for s in shots] == listing_planner.DEFAULT_SHOT_TYPES[:3]
    assert all("prompt" in s for s in shots)


def test_json_array_missing_required_keys_falls_back_to_default_shots(monkeypatch):
    """Well-formed JSON, but shaped wrong (missing "prompt") -- must not
    silently submit a job with no prompt at all."""
    bad_json = json.dumps([{"shot_type": "hero"}])
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": bad_json}))

    shots = listing_planner.plan_listing_shots(_tool(), [], None, output_count=2)

    assert [s["shot_type"] for s in shots] == listing_planner.DEFAULT_SHOT_TYPES[:2]


def test_empty_json_array_falls_back_to_default_shots(monkeypatch):
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": "[]"}))

    shots = listing_planner.plan_listing_shots(_tool(), [], None, output_count=2)

    assert [s["shot_type"] for s in shots] == listing_planner.DEFAULT_SHOT_TYPES[:2]


def test_json_object_instead_of_array_falls_back_to_default_shots(monkeypatch):
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": '{"shot_type": "hero", "prompt": "x"}'}))

    shots = listing_planner.plan_listing_shots(_tool(), [], None, output_count=1)

    assert [s["shot_type"] for s in shots] == listing_planner.DEFAULT_SHOT_TYPES[:1]


# --------------------------------------------------------------------------- #
# model returning too many/too few shots -- trimmed/padded to exactly
# output_count
# --------------------------------------------------------------------------- #

def test_too_many_shots_are_trimmed_to_exactly_output_count(monkeypatch):
    shots_json = json.dumps([
        {"shot_type": "hero", "prompt": "a"},
        {"shot_type": "side", "prompt": "b"},
        {"shot_type": "wearing", "prompt": "c"},
        {"shot_type": "sole", "prompt": "d"},
    ])
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": shots_json}))

    shots = listing_planner.plan_listing_shots(_tool(), [], None, output_count=2)

    assert len(shots) == 2
    assert [s["shot_type"] for s in shots] == ["hero", "side"]


def test_too_few_shots_are_padded_to_exactly_output_count(monkeypatch):
    shots_json = json.dumps([{"shot_type": "hero", "prompt": "a"}])
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": shots_json}))

    shots = listing_planner.plan_listing_shots(_tool(), [], None, output_count=4)

    assert len(shots) == 4
    assert shots[0] == {"shot_type": "hero", "prompt": "a"}
    # the remaining 3 are padded from the fallback shot types, in order
    assert [s["shot_type"] for s in shots[1:]] == listing_planner.DEFAULT_SHOT_TYPES[:3]


def test_exact_count_is_returned_unchanged(monkeypatch):
    shots_json = json.dumps([{"shot_type": t, "prompt": t} for t in listing_planner.DEFAULT_SHOT_TYPES])
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": shots_json}))

    shots = listing_planner.plan_listing_shots(_tool(), [], None, output_count=len(listing_planner.DEFAULT_SHOT_TYPES))

    assert len(shots) == len(listing_planner.DEFAULT_SHOT_TYPES)


# --------------------------------------------------------------------------- #
# ai_steps.shot_planner overrides -- model_id/model/system_prompt all
# configurable per tool_definitions row
# --------------------------------------------------------------------------- #

def test_defaults_used_when_ai_steps_has_no_shot_planner_entry(monkeypatch):
    call_fal_sync = MagicMock(return_value={"output": json.dumps([{"shot_type": "hero", "prompt": "x"}])})
    monkeypatch.setattr(listing_planner, "call_fal_sync", call_fal_sync)

    listing_planner.plan_listing_shots(_tool(ai_steps={}), [], None, output_count=1)

    kwargs = call_fal_sync.call_args.kwargs
    assert kwargs["model_id"] == "openrouter/router/vision"
    assert kwargs["input_params"]["model"] == "google/gemini-2.5-flash"
    assert kwargs["input_params"]["system_prompt"] == listing_planner._DEFAULT_SYSTEM_PROMPT


def test_custom_fallback_shot_prompt_template_in_ai_steps_is_used_on_malformed_json(monkeypatch):
    """ai_steps.fallback_shot_prompt_template is DB-configurable -- must be
    used instead of the hardcoded default when the model's output can't be
    parsed at all."""
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": "not json"}))

    shots = listing_planner.plan_listing_shots(
        _tool(ai_steps={"shot_planner": {**SHOT_PLANNER_STEP, "fallback_shot_prompt_template": "A {shot_type} shot, custom style"}}),
        [], None, output_count=2,
    )

    assert shots[0]["prompt"] == f"A {listing_planner.DEFAULT_SHOT_TYPES[0]} shot, custom style"


def test_custom_fallback_shot_prompt_template_is_used_when_padding_too_few_shots(monkeypatch):
    shots_json = json.dumps([{"shot_type": "hero", "prompt": "a"}])
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": shots_json}))

    shots = listing_planner.plan_listing_shots(
        _tool(ai_steps={"shot_planner": {**SHOT_PLANNER_STEP, "fallback_shot_prompt_template": "Custom {shot_type} padding"}}),
        [], None, output_count=2,
    )

    assert shots[1]["prompt"] == f"Custom {listing_planner.DEFAULT_SHOT_TYPES[0]} padding"


def test_default_fallback_shot_prompt_template_used_when_not_configured(monkeypatch):
    monkeypatch.setattr(listing_planner, "call_fal_sync", MagicMock(return_value={"output": "not json"}))

    shots = listing_planner.plan_listing_shots(_tool(ai_steps={}), [], None, output_count=1)

    assert shots[0]["prompt"] == listing_planner._DEFAULT_FALLBACK_SHOT_PROMPT_TEMPLATE.format(
        shot_type=listing_planner.DEFAULT_SHOT_TYPES[0]
    )


def test_custom_system_prompt_in_ai_steps_is_used(monkeypatch):
    call_fal_sync = MagicMock(return_value={"output": json.dumps([{"shot_type": "hero", "prompt": "x"}])})
    monkeypatch.setattr(listing_planner, "call_fal_sync", call_fal_sync)

    listing_planner.plan_listing_shots(
        _tool(ai_steps={"shot_planner": {**SHOT_PLANNER_STEP, "system_prompt": "custom instructions"}}),
        [], None, output_count=1,
    )

    assert call_fal_sync.call_args.kwargs["input_params"]["system_prompt"] == "custom instructions"
