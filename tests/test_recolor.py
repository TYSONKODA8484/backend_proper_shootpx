"""app/tools/recolor.py::build_instruction -- the two Recolor paths:
blank target_area (vision call to detect the target) vs. a given target_area
(pure template, no AI call)."""

from unittest.mock import MagicMock

from app.tools import recolor


def _job(color="red", target_area=None, image_urls=None):
    return MagicMock(input_params={
        "color": color,
        "target_area": target_area,
        "image_urls": image_urls if image_urls is not None else ["https://fal.test/uploaded.png"],
    })


def _tool_definition(ai_steps=None):
    return MagicMock(ai_steps=ai_steps if ai_steps is not None else {})


VISION_STEP = {
    "model": "fal-ai/moondream-next",
    "task_type": "detection",
    "prompt_template": "Find the main product and recolor it to {color}",
}


# --------------------------------------------------------------------------- #
# target_area given -- template path, no AI call
# --------------------------------------------------------------------------- #

def test_given_target_area_builds_instruction_directly_without_a_vision_call(monkeypatch):
    call_fal_sync = MagicMock()
    monkeypatch.setattr(recolor, "call_fal_sync", call_fal_sync)

    job = _job(color="blue", target_area="left sleeve")
    tool = _tool_definition(ai_steps={"detect_target": VISION_STEP})

    instruction = recolor.build_instruction(job, tool)

    assert instruction == "Recolor the left sleeve to color blue"
    call_fal_sync.assert_not_called()


# --------------------------------------------------------------------------- #
# target_area blank -- vision path, calls the configured detect_target model
# --------------------------------------------------------------------------- #

def test_blank_target_area_calls_the_configured_vision_model(monkeypatch):
    call_fal_sync = MagicMock(return_value={"answer": "Recolor the sneaker sole to color red"})
    monkeypatch.setattr(recolor, "call_fal_sync", call_fal_sync)

    job = _job(color="red", target_area=None, image_urls=["https://fal.test/first.png", "https://fal.test/second.png"])
    tool = _tool_definition(ai_steps={"detect_target": VISION_STEP})

    instruction = recolor.build_instruction(job, tool)

    assert instruction == "Recolor the sneaker sole to color red"
    call_fal_sync.assert_called_once_with(
        model_id="fal-ai/moondream-next",
        input_params={
            "task_type": "detection",
            "image_url": "https://fal.test/first.png",  # first image, even with multiple uploaded
            "query": "Find the main product and recolor it to red",
        },
    )


def test_blank_target_area_falls_back_to_a_flat_instruction_when_vision_gives_no_answer(monkeypatch):
    monkeypatch.setattr(recolor, "call_fal_sync", MagicMock(return_value={}))

    job = _job(color="green", target_area=None)
    tool = _tool_definition(ai_steps={"detect_target": VISION_STEP})

    instruction = recolor.build_instruction(job, tool)

    assert instruction == "Recolor the main product to color green"


def test_blank_target_area_skips_the_vision_call_when_no_detect_target_step_configured(monkeypatch):
    """Defensive fallback -- should never happen if tool_definitions is
    configured correctly, but must never crash the job over missing config."""
    call_fal_sync = MagicMock()
    monkeypatch.setattr(recolor, "call_fal_sync", call_fal_sync)

    job = _job(color="black", target_area=None)
    tool = _tool_definition(ai_steps={})  # no detect_target entry

    instruction = recolor.build_instruction(job, tool)

    assert instruction == "Recolor the main product to color black"
    call_fal_sync.assert_not_called()


def test_blank_target_area_with_no_uploaded_images_still_calls_vision_with_null_url(monkeypatch):
    """Should never happen in practice (images are required on /generate), but
    must not raise an IndexError if input_params ever arrives with an empty list."""
    call_fal_sync = MagicMock(return_value={"answer": "fallback answer"})
    monkeypatch.setattr(recolor, "call_fal_sync", call_fal_sync)

    job = _job(color="red", target_area=None, image_urls=[])
    tool = _tool_definition(ai_steps={"detect_target": VISION_STEP})

    recolor.build_instruction(job, tool)

    assert call_fal_sync.call_args.kwargs["input_params"]["image_url"] is None
