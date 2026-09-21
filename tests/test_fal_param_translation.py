"""app/worker.py::_translate_fal_params -- Recolor moved to fal-ai/
flux-2/edit; its own param_schema fields ("color", "target_area",
"aspect_ratio", "resolution") are UI-only and must never be forwarded raw.
image_size for recolor is computed by the caller (recolor.build_image_size,
from ai_steps.size_map) BEFORE _translate_fal_params runs, so this module
only has to strip aspect_ratio/resolution and leave the already-built
image_size untouched -- see tests/test_recolor.py for size_map coverage.

creative_photoshoot and listing_photoshoot both follow the exact same pattern
(their own aspect_ratio/resolution/quality param_schema fields, resolved by
the caller via resolve_generation_params BEFORE this module runs -- see
tests/test_creative_photoshoot.py and test_generation.py's listing_photoshoot
pricing section). Confirmed against
https://fal.ai/models/openai/gpt-image-2/edit/api:
  quality (enum):     auto | low | medium | high
  image_size (enum):  square_hd | square | portrait_4_3 | portrait_16_9 |
                       landscape_4_3 | landscape_16_9 | auto
  image_size (object): {width, height} -- both multiples of 16, max edge
                       3840px, aspect ratio <= 3:1, total pixels between
                       655,360 and 8,294,400.
"""

import pytest

from app import worker

# The real, complete accepted-value set from fal's own docs.
FAL_REAL_QUALITY_VALUES = {"auto", "low", "medium", "high"}
FAL_REAL_IMAGE_SIZE_PRESETS = {
    "square_hd", "square", "portrait_4_3", "portrait_16_9",
    "landscape_4_3", "landscape_16_9", "auto",
}


def _is_valid_fal_image_size(value) -> bool:
    if isinstance(value, str):
        return value in FAL_REAL_IMAGE_SIZE_PRESETS
    if isinstance(value, dict):
        width, height = value.get("width"), value.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            return False
        if width % 16 != 0 or height % 16 != 0:
            return False
        if max(width, height) > 3840:
            return False
        if max(width, height) / min(width, height) > 3:
            return False
        total = width * height
        return 655_360 <= total <= 8_294_400
    return False


# --------------------------------------------------------------------------- #
# quality: gpt-image-2/edit's own real values (auto/low/medium/high) pass
# through completely unchanged when no feature-scoped translation applies.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", ["auto", "low", "medium", "high"])
def test_every_gpt_image_2_edit_quality_value_passes_through_unchanged(value):
    result = worker._translate_fal_params({"quality": value})
    assert result["quality"] == value
    assert result["quality"] in FAL_REAL_QUALITY_VALUES


def test_unrecognized_quality_value_still_passes_through_unchanged():
    """Defensive: an unexpected value must not silently disappear or crash --
    it passes through so a real fal 422 makes the gap obvious."""
    result = worker._translate_fal_params({"quality": "ultra-mega"})
    assert result["quality"] == "ultra-mega"


# --------------------------------------------------------------------------- #
# size: our aspect-ratio strings -> fal's real image_size presets/object
# (creative_photoshoot's own "size" field, gpt-image-2/edit's real schema)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("our_value", [
    "original", "1:1", "9:16", "3:4", "2:3", "3:2", "4:3", "16:9",
])
def test_every_size_option_translates_to_a_genuinely_valid_fal_image_size(our_value):
    result = worker._translate_fal_params({"size": our_value})
    assert "size" not in result          # renamed, not left behind under the old key
    assert "image_size" in result
    assert _is_valid_fal_image_size(result["image_size"]), (
        f"size={our_value!r} translated to {result['image_size']!r}, "
        f"which is not a value fal's image_size parameter actually accepts"
    )


def test_size_is_renamed_to_image_size_not_sent_under_both_keys():
    result = worker._translate_fal_params({"size": "1:1", "image_urls": ["https://fal.test/1.png"]})
    assert result == {"image_size": "square_hd", "image_urls": ["https://fal.test/1.png"]}


def test_original_maps_to_fals_auto_default():
    assert worker._translate_fal_params({"size": "original"})["image_size"] == "auto"


def test_unrecognized_size_value_passes_through_under_the_renamed_key():
    result = worker._translate_fal_params({"size": "21:9"})
    assert result["image_size"] == "21:9"


# --------------------------------------------------------------------------- #
# The original dict must never be mutated -- job.input_params keeps the
# user-facing values for display and (at creation time) credit-cost lookup.
# --------------------------------------------------------------------------- #

def test_translation_does_not_mutate_the_input_dict():
    original = {"quality": "high", "size": "16:9", "color": "red"}
    snapshot = dict(original)

    worker._translate_fal_params(original)

    assert original == snapshot


def test_color_and_target_area_are_stripped_not_forwarded_to_fal():
    """color/target_area are Recolor's own UI fields, already consumed by
    build_instruction() into the `prompt` string -- not a real field on any
    fal model this backend calls, must never be forwarded raw."""
    result = worker._translate_fal_params({"color": "red", "target_area": "sneaker sole", "quality": "low"})
    assert "color" not in result
    assert "target_area" not in result
    assert result["quality"] == "low"


# --------------------------------------------------------------------------- #
# recolor: fal-ai/flux-2/edit. image_size is computed by the caller BEFORE
# _translate_fal_params runs (recolor.build_image_size, off ai_steps.size_map
# -- see tests/test_recolor.py); this module's only job for recolor is
# stripping aspect_ratio/resolution (its own UI fields, not real flux-2/edit
# input fields) and leaving the already-built image_size untouched.
# --------------------------------------------------------------------------- #

def test_recolors_aspect_ratio_and_resolution_are_stripped_not_forwarded_to_fal():
    result = worker._translate_fal_params(
        {"aspect_ratio": "16:9", "resolution": "high", "image_size": {"width": 1920, "height": 1080}},
        feature_type="recolor",
    )
    assert "aspect_ratio" not in result
    assert "resolution" not in result
    assert result["image_size"] == {"width": 1920, "height": 1080}


def test_recolors_aspect_ratio_and_resolution_stripping_is_scoped_to_recolor_only():
    """Must never strip these for a different tool that happens to reuse the
    field names (none currently do, but this is the safeguard)."""
    result = worker._translate_fal_params({"aspect_ratio": "16:9", "resolution": "high"})
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "high"


def test_full_recolor_payload_translates_cleanly():
    """image_size here stands in for what submit_generation_to_fal actually
    sets before calling _translate_fal_params (recolor.build_image_size's
    real output) -- this test only covers _translate_fal_params's own job of
    stripping recolor's UI-only fields."""
    params = {
        "color": "red",
        "target_area": "sneaker sole",
        "aspect_ratio": "9:16",
        "resolution": "high",
        "image_size": {"width": 1080, "height": 1920},
        "image_urls": ["https://fal.test/1.png"],
        "prompt": "Recolor the sneaker sole to color red",
    }

    result = worker._translate_fal_params(params, feature_type="recolor")

    assert result == {
        "image_size": {"width": 1080, "height": 1920},
        "image_urls": ["https://fal.test/1.png"],
        "prompt": "Recolor the sneaker sole to color red",
    }


# --------------------------------------------------------------------------- #
# creative_photoshoot: fal-ai/openai/gpt-image-2.5/flare/edit. image_size AND
# quality are computed by the caller BEFORE _translate_fal_params runs
# (creative_photoshoot.resolve_generation_params, off ai_steps.size_map +
# quality_multiplier/resolution_multiplier -- see tests/test_creative_photoshoot.py),
# same pattern as recolor. This module's only job for creative_photoshoot is
# stripping "idea" (already consumed into the prompt by build_instruction())
# and aspect_ratio/resolution (its own UI fields, not real fal input fields)
# and leaving the already-built image_size/quality untouched.
# --------------------------------------------------------------------------- #

def test_idea_is_stripped_not_forwarded_to_fal():
    """"idea" is creative_photoshoot's own UI/schema field, already consumed
    by build_instruction() into the `prompt` string -- the real fal schema has
    no such field."""
    result = worker._translate_fal_params({"idea": "Sci-Fi", "quality": "medium"})
    assert "idea" not in result
    assert result["quality"] == "medium"


def test_creative_photoshoots_aspect_ratio_and_resolution_are_stripped_not_forwarded_to_fal():
    result = worker._translate_fal_params(
        {"aspect_ratio": "16:9", "resolution": "4k", "image_size": {"width": 3840, "height": 2160}, "quality": "high"},
        feature_type="creative_photoshoot",
    )
    assert "aspect_ratio" not in result
    assert "resolution" not in result
    assert result["image_size"] == {"width": 3840, "height": 2160}
    assert result["quality"] == "high"


def test_creative_photoshoots_aspect_ratio_and_resolution_stripping_is_scoped_to_creative_photoshoot_only():
    """Must never strip these for a different tool that happens to reuse the
    field names."""
    result = worker._translate_fal_params({"aspect_ratio": "16:9", "resolution": "4k"})
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "4k"


# --------------------------------------------------------------------------- #
# listing_photoshoot: same pattern as creative_photoshoot above -- image_size
# AND quality are computed by the caller BEFORE _translate_fal_params runs
# (listing_planner.resolve_generation_params, off ai_steps.size_map +
# quality_multiplier/resolution_multiplier -- see tests/test_generation.py's
# listing_photoshoot pricing section). This module's only job here is
# stripping aspect_ratio/resolution (its own UI fields, not real fal input
# fields) and leaving the already-built image_size/quality untouched.
# --------------------------------------------------------------------------- #

def test_listing_photoshoots_aspect_ratio_and_resolution_are_stripped_not_forwarded_to_fal():
    result = worker._translate_fal_params(
        {"aspect_ratio": "16:9", "resolution": "4k", "image_size": {"width": 3840, "height": 2160}, "quality": "high"},
        feature_type="listing_photoshoot",
    )
    assert "aspect_ratio" not in result
    assert "resolution" not in result
    assert result["image_size"] == {"width": 3840, "height": 2160}
    assert result["quality"] == "high"


def test_listing_photoshoots_aspect_ratio_and_resolution_stripping_is_scoped_to_listing_photoshoot_only():
    """Must never strip these for a different tool that happens to reuse the
    field names."""
    result = worker._translate_fal_params({"aspect_ratio": "16:9", "resolution": "4k"})
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "4k"


def test_listing_photoshoots_shot_type_is_stripped_not_forwarded_to_fal():
    """"shot_type" is per-job planning metadata written by plan_listing_shots'
    per_job_overrides -- not part of the real fal schema."""
    result = worker._translate_fal_params(
        {"shot_type": "hero", "quality": "medium"}, feature_type="listing_photoshoot",
    )
    assert "shot_type" not in result
    assert result["quality"] == "medium"


# --------------------------------------------------------------------------- #
# model_shoot: same pattern as recolor/creative_photoshoot/listing_photoshoot
# above -- image_size is computed by the caller BEFORE _translate_fal_params
# runs (model_shoot.resolve_generation_params, off ai_steps.size_map -- see
# tests/test_generation.py's model_shoot pricing section and
# tests/test_fal_integration.py's submission tests). This module's only job
# for model_shoot is stripping aspect_ratio/resolution (its own UI fields,
# not real seedream v4.5/edit input fields) and recombining the 3 image
# categories into the single "image_urls" list seedream's real schema
# actually has.
# --------------------------------------------------------------------------- #

def test_model_shoots_aspect_ratio_and_resolution_are_stripped_not_forwarded_to_fal():
    result = worker._translate_fal_params(
        {"aspect_ratio": "16:9", "resolution": "4k", "image_size": {"width": 4096, "height": 2304}},
        feature_type="model_shoot",
    )
    assert "aspect_ratio" not in result
    assert "resolution" not in result
    assert result["image_size"] == {"width": 4096, "height": 2304}


def test_model_shoots_aspect_ratio_and_resolution_stripping_is_scoped_to_model_shoot_only():
    """Must never strip these for a different tool that happens to reuse the
    field names."""
    result = worker._translate_fal_params({"aspect_ratio": "16:9", "resolution": "4k"})
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "4k"


def test_model_shoots_garment_groups_are_recombined_into_one_image_urls_list_in_order():
    """seedream's real image field is a single "image_urls" list (schema:
    prompt + image_urls, both required, max 10 -- matching model_shoot's own
    max_input_images=10 exactly). model_image/garments/reference_images only
    exist so plan_model_shoot() can label each image differently in its
    vision prompt -- the real generation call needs them recombined in the
    same order the planner labeled them: model reference first, then each
    garment group's images (in group order, then image order within the
    group), then style/pose references."""
    result = worker._translate_fal_params(
        {
            "model_image": "https://fal.test/model.png",
            "garments": [
                {"label": "Top", "image_urls": ["https://fal.test/shirt-front.png", "https://fal.test/shirt-back.png"]},
                {"label": "Bottom", "image_urls": ["https://fal.test/pants.png"]},
            ],
            "reference_images": ["https://fal.test/pose.png"],
            "image_size": {"width": 1920, "height": 1920},
        },
        feature_type="model_shoot",
    )
    assert result["image_urls"] == [
        "https://fal.test/model.png",
        "https://fal.test/shirt-front.png",
        "https://fal.test/shirt-back.png",
        "https://fal.test/pants.png",
        "https://fal.test/pose.png",
    ]
    assert "model_image" not in result
    assert "garments" not in result
    assert "reference_images" not in result


def test_model_shoots_image_recombination_handles_no_garment_or_reference_images():
    result = worker._translate_fal_params(
        {"model_image": "https://fal.test/model.png", "image_size": {"width": 1920, "height": 1920}},
        feature_type="model_shoot",
    )
    assert result["image_urls"] == ["https://fal.test/model.png"]


# --------------------------------------------------------------------------- #
# model_shoot_generate_model: structured attributes are already consumed into
# the AI-written prompt by build_instruction() -- none of them are part of
# openai/gpt-image-2's real schema (prompt/image_size/background/quality/
# num_images/output_format/sync_mode -- confirmed against
# https://fal.ai/models/openai/gpt-image-2/api). This model also has NO
# image_urls field at all (it's the plain text-to-image endpoint, not
# /edit) -- unlike every other tool, image_urls must be stripped here too,
# not forwarded even as an empty list.
# --------------------------------------------------------------------------- #

def test_model_shoot_generate_models_structured_attributes_are_stripped():
    result = worker._translate_fal_params(
        {
            "gender": "Female", "age_bracket": "Adult (30s-40s)", "ethnicity": "South Asian",
            "skin_tone": "Fair", "body_type": "Slim", "notes": "warm smile",
            "prompt": "A professional studio portrait.",
        },
        feature_type="model_shoot_generate_model",
    )
    assert result == {"prompt": "A professional studio portrait."}


def test_model_shoot_generate_models_empty_image_urls_is_stripped_not_forwarded():
    """Real gap: /generate's generic path always sets input_params["image_urls"]
    to a list (empty here, since this tool's max_input_images=0 blocks any
    upload) -- openai/gpt-image-2 (plain text-to-image) has no such field at
    all in its real schema, so it must never be forwarded, not even empty."""
    result = worker._translate_fal_params(
        {"prompt": "A professional studio portrait.", "image_urls": []},
        feature_type="model_shoot_generate_model",
    )
    assert "image_urls" not in result


def test_model_shoot_generate_models_field_stripping_does_not_leak_to_other_tools():
    """Structured-attribute field names are unique to this tool, but confirm
    the image_urls strip specifically is scoped -- every other tool's real
    image_urls (even an empty list, e.g. enhance_prompt) must still pass
    through untouched."""
    result = worker._translate_fal_params({"prompt": "x", "image_urls": []}, feature_type="enhance_prompt")
    assert result["image_urls"] == []
