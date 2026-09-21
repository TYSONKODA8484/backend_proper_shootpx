"""app/tools/recolor.py::build_instruction -- the two Recolor paths:
blank target_area (vision call to detect the target) vs. a given target_area
(pure template, no AI call)."""

from unittest.mock import MagicMock

from app.tools import recolor
from tests.prompt_fixtures import ai_steps_for


def _job(color="red", target_area=None, image_urls=None, aspect_ratio=None, resolution=None):
    return MagicMock(
        id="job-1",
        feature_type="recolor",
        input_params={
            "color": color,
            "target_area": target_area,
            "image_urls": image_urls if image_urls is not None else ["https://fal.test/uploaded.png"],
            **({"aspect_ratio": aspect_ratio} if aspect_ratio is not None else {}),
            **({"resolution": resolution} if resolution is not None else {}),
        },
    )


def _tool_definition(ai_steps=None):
    """Defaults to a fully-migrated recolor row -- every prompt string this
    tool uses now comes from ai_steps, never from Python."""
    return MagicMock(ai_steps=ai_steps if ai_steps is not None else ai_steps_for("recolor"))


VISION_STEP = ai_steps_for("recolor")["detect_target"]


# --------------------------------------------------------------------------- #
# target_area given -- template path, no AI call
# --------------------------------------------------------------------------- #

def test_given_target_area_builds_instruction_directly_without_a_vision_call(monkeypatch):
    call_fal_sync = MagicMock()
    monkeypatch.setattr(recolor, "call_fal_sync", call_fal_sync)

    job = _job(color="blue", target_area="left sleeve")
    tool = _tool_definition()

    instruction = recolor.build_instruction(job, tool, MagicMock())

    assert instruction == "Recolor the left sleeve to color blue"
    call_fal_sync.assert_not_called()


# --------------------------------------------------------------------------- #
# target_area blank -- vision path, calls the configured detect_target model
# --------------------------------------------------------------------------- #

def test_blank_target_area_calls_the_configured_vision_model(monkeypatch):
    """moondream-next's real schema (confirmed against
    https://fal.ai/models/fal-ai/moondream-next/api): the prompt field is
    named `prompt` (not `query`), and the response comes back as
    {"output": ...} (not {"answer": ...})."""
    call_fal_sync = MagicMock(return_value={"output": "Recolor the sneaker sole to color red"})
    monkeypatch.setattr(recolor, "call_fal_sync", call_fal_sync)

    job = _job(color="red", target_area=None, image_urls=["https://fal.test/first.png", "https://fal.test/second.png"])
    tool = _tool_definition()

    instruction = recolor.build_instruction(job, tool, MagicMock())

    assert instruction == "Recolor the sneaker sole to color red"
    call_fal_sync.assert_called_once_with(
        model_id="fal-ai/moondream-next",
        input_params={
            "task_type": "query",
            "image_url": "https://fal.test/first.png",  # first image, even with multiple uploaded
            "prompt": "Find the main product and recolor it to red",
        },
    )


def test_blank_target_area_falls_back_to_a_flat_instruction_when_vision_gives_no_answer(monkeypatch):
    monkeypatch.setattr(recolor, "call_fal_sync", MagicMock(return_value={}))

    job = _job(color="green", target_area=None)
    tool = _tool_definition()

    instruction = recolor.build_instruction(job, tool, MagicMock())

    assert instruction == "Recolor the main product to color green"


def test_blank_target_area_skips_the_vision_call_when_no_detect_target_step_configured(monkeypatch):
    """Defensive fallback -- should never happen if tool_definitions is
    configured correctly, but must never crash the job over missing config."""
    call_fal_sync = MagicMock()
    monkeypatch.setattr(recolor, "call_fal_sync", call_fal_sync)

    job = _job(color="black", target_area=None)
    tool = _tool_definition(ai_steps={
        k: v for k, v in ai_steps_for("recolor").items() if k != "detect_target"
    })  # no detect_target entry

    instruction = recolor.build_instruction(job, tool, MagicMock())

    assert instruction == "Recolor the main product to color black"
    call_fal_sync.assert_not_called()


def test_blank_target_area_with_no_uploaded_images_still_calls_vision_with_null_url(monkeypatch):
    """Should never happen in practice (images are required on /generate), but
    must not raise an IndexError if input_params ever arrives with an empty list."""
    call_fal_sync = MagicMock(return_value={"output": "fallback answer"})
    monkeypatch.setattr(recolor, "call_fal_sync", call_fal_sync)

    job = _job(color="red", target_area=None, image_urls=[])
    tool = _tool_definition()

    recolor.build_instruction(job, tool, MagicMock())

    assert call_fal_sync.call_args.kwargs["input_params"]["image_url"] is None


# --------------------------------------------------------------------------- #
# build_image_size -- ai_steps.size_map lookup by aspect_ratio + resolution,
# the {width, height} object sent to fal-ai/flux-2/edit's image_size field.
# --------------------------------------------------------------------------- #

def test_build_image_size_looks_up_the_exact_size_map_entry():
    job = _job(aspect_ratio="16:9", resolution="high")
    tool = _tool_definition()

    size = recolor.build_image_size(job, tool)

    assert size == {"width": 1920, "height": 1080}


def test_build_image_size_standard_and_high_resolution_differ():
    """The whole point of the resolution field: "standard" and "high" must
    resolve to genuinely different image sizes for the same aspect ratio --
    otherwise picking "high" (2 credits) buys nothing over "standard" (1)."""
    tool = _tool_definition()

    standard = recolor.build_image_size(_job(aspect_ratio="1:1", resolution="standard"), tool)
    high = recolor.build_image_size(_job(aspect_ratio="1:1", resolution="high"), tool)

    assert standard == {"width": 1024, "height": 1024}
    assert high == {"width": 1408, "height": 1408}
    assert standard != high


def test_build_image_size_defaults_when_aspect_ratio_and_resolution_are_missing():
    job = _job(aspect_ratio=None, resolution=None)
    tool = _tool_definition()

    size = recolor.build_image_size(job, tool)

    assert size == {"width": 1024, "height": 1024}  # 1:1 / standard defaults


def test_build_image_size_falls_back_to_1_1_for_an_unknown_aspect_ratio():
    job = _job(aspect_ratio="21:9", resolution="high")
    tool = _tool_definition()

    size = recolor.build_image_size(job, tool)

    assert size == {"width": 1408, "height": 1408}  # 1:1 / high


def test_build_image_size_falls_back_to_standard_for_an_unknown_resolution():
    job = _job(aspect_ratio="9:16", resolution="ultra")
    tool = _tool_definition()

    size = recolor.build_image_size(job, tool)

    assert size == {"width": 768, "height": 1360}  # 9:16 / standard


def test_build_image_size_falls_back_to_hardcoded_default_when_no_size_map_configured():
    job = _job(aspect_ratio="1:1", resolution="high")
    tool = _tool_definition(ai_steps={
        k: v for k, v in ai_steps_for("recolor").items() if k != "size_map"
    })

    size = recolor.build_image_size(job, tool)

    assert size == {"width": 1024, "height": 1024}
