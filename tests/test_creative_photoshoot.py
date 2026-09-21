"""app/tools/creative_photoshoot.py::build_instruction -- three paths:
idea-only / idea+prompt (vision call to build the scene description),
prompt-only (pure passthrough, no AI call), and the missing-both error."""

from unittest.mock import MagicMock

import pytest

from app.tools import creative_photoshoot
from tests.prompt_fixtures import ai_steps_for


def _job(idea=None, prompt=None, image_urls=None):
    return MagicMock(input_params={
        "idea": idea,
        "prompt": prompt,
        "image_urls": image_urls if image_urls is not None else ["https://fal.test/uploaded.png"],
    })


def _tool_definition(ai_steps=None):
    """Defaults to a fully-migrated row -- all prompt text comes from ai_steps."""
    return MagicMock(ai_steps=ai_steps if ai_steps is not None else ai_steps_for("creative_photoshoot"))


SCENE_VISION_STEP = ai_steps_for("creative_photoshoot")["scene_vision"]


# --------------------------------------------------------------------------- #
# prompt only -- pure passthrough, no AI call
# --------------------------------------------------------------------------- #

def test_prompt_only_returns_the_prompt_untouched_without_any_vision_call(monkeypatch):
    call_fal_sync = MagicMock()
    monkeypatch.setattr(creative_photoshoot, "call_fal_sync", call_fal_sync)

    job = _job(idea=None, prompt="a sneaker floating in zero gravity")
    tool = _tool_definition()

    instruction = creative_photoshoot.build_instruction(job, tool, MagicMock())

    assert instruction == "a sneaker floating in zero gravity"
    call_fal_sync.assert_not_called()


# --------------------------------------------------------------------------- #
# idea only -- vision path, calls the configured scene_vision model
# --------------------------------------------------------------------------- #

def test_idea_only_calls_the_configured_vision_model(monkeypatch):
    call_fal_sync = MagicMock(return_value={"output": "A sneaker glowing under neon city lights, cinematic."})
    monkeypatch.setattr(creative_photoshoot, "call_fal_sync", call_fal_sync)

    job = _job(idea="Sci-Fi", prompt=None, image_urls=["https://fal.test/first.png", "https://fal.test/second.png"])
    tool = _tool_definition()

    instruction = creative_photoshoot.build_instruction(job, tool, MagicMock())

    assert instruction == "A sneaker glowing under neon city lights, cinematic."
    call_fal_sync.assert_called_once_with(
        model_id="openrouter/router/vision",
        input_params={
            "image_urls": ["https://fal.test/first.png", "https://fal.test/second.png"],  # ALL uploaded images, not just the first
            "prompt": "Describe this product photo as a creative scene for the following idea: Sci-Fi. "
                      "Write it as a single, concrete, visual scene description suitable as an image generation prompt.",
            "model": "google/gemini-2.5-flash",
        },
    )


# --------------------------------------------------------------------------- #
# idea + prompt -- vision path, but the user's own prompt is appended to the
# vision query so it can steer the generated scene description
# --------------------------------------------------------------------------- #

def test_idea_and_prompt_appends_the_users_prompt_to_the_vision_query(monkeypatch):
    call_fal_sync = MagicMock(return_value={"output": "A sneaker on a marble pedestal under gold light."})
    monkeypatch.setattr(creative_photoshoot, "call_fal_sync", call_fal_sync)

    job = _job(idea="Luxury", prompt="add gold accents")
    tool = _tool_definition()

    instruction = creative_photoshoot.build_instruction(job, tool, MagicMock())

    assert instruction == "A sneaker on a marble pedestal under gold light."
    sent_prompt = call_fal_sync.call_args.kwargs["input_params"]["prompt"]
    assert sent_prompt.endswith("The user also specifically asked for: add gold accents")


def test_idea_and_prompt_falls_back_to_idea_plus_prompt_when_vision_gives_no_answer(monkeypatch):
    monkeypatch.setattr(creative_photoshoot, "call_fal_sync", MagicMock(return_value={}))

    job = _job(idea="Luxury", prompt="add gold accents")
    tool = _tool_definition()

    instruction = creative_photoshoot.build_instruction(job, tool, MagicMock())

    assert instruction == "Luxury. add gold accents"


def test_idea_only_falls_back_to_idea_when_vision_gives_no_answer(monkeypatch):
    monkeypatch.setattr(creative_photoshoot, "call_fal_sync", MagicMock(return_value={}))

    job = _job(idea="Sci-Fi", prompt=None)
    tool = _tool_definition()

    instruction = creative_photoshoot.build_instruction(job, tool, MagicMock())

    assert instruction == "Sci-Fi"


def test_idea_given_skips_the_vision_call_when_no_scene_vision_step_configured(monkeypatch):
    """Defensive fallback -- should never happen if tool_definitions is
    configured correctly, but must never crash the job over missing config."""
    call_fal_sync = MagicMock()
    monkeypatch.setattr(creative_photoshoot, "call_fal_sync", call_fal_sync)

    job = _job(idea="Sci-Fi", prompt="add fog")
    tool = _tool_definition(ai_steps={
        k: v for k, v in ai_steps_for("creative_photoshoot").items() if k != "scene_vision"
    })  # no scene_vision entry

    instruction = creative_photoshoot.build_instruction(job, tool, MagicMock())

    assert instruction == "Sci-Fi. add fog"
    call_fal_sync.assert_not_called()


# --------------------------------------------------------------------------- #
# missing both -- must raise, never silently submit an empty prompt
# --------------------------------------------------------------------------- #

def test_raises_when_both_idea_and_prompt_are_missing():
    job = _job(idea=None, prompt=None)
    tool = _tool_definition()

    with pytest.raises(ValueError, match="needs at least an idea or a prompt"):
        creative_photoshoot.build_instruction(job, tool, MagicMock())
