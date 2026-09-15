"""app/worker.py::_translate_fal_params -- Recolor's param_schema stores
"size" as our own plain aspect-ratio strings, which must never be sent to
fal.ai as-is; "quality" is now stored as fal's own real enum values directly
(auto/low/medium/high), so it passes through completely untouched. Confirmed
against fal's real schema at https://fal.ai/models/openai/gpt-image-2/edit/api:
  quality (enum):     auto | low | medium | high
  image_size (enum):  square_hd | square | portrait_4_3 | portrait_16_9 |
                       landscape_4_3 | landscape_16_9 | auto
  image_size (object): {width, height} -- both multiples of 16, max edge
                       3840px, aspect ratio <= 3:1, total pixels between
                       655,360 and 8,294,400.
"""

import pytest

from app import worker

# The real, complete accepted-value set from fal's own docs -- if the
# quality options stored in recolor's param_schema are ever edited to
# something outside this set, that's a real 422 waiting to happen.
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
# quality: stored directly as fal's own real values now (auto/low/medium/
# high) -- no translation table, passes through completely unchanged.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", ["auto", "low", "medium", "high"])
def test_every_quality_option_in_recolors_actual_schema_passes_through_unchanged(value):
    """Cross-check against the literal options our /tools/recolor/schema
    endpoint now serves -- every value the frontend can possibly send is
    already one fal genuinely accepts, so _translate_fal_params must not
    alter it at all."""
    result = worker._translate_fal_params({"quality": value})
    assert result["quality"] == value
    assert result["quality"] in FAL_REAL_QUALITY_VALUES


def test_unrecognized_quality_value_still_passes_through_unchanged():
    """Defensive: an unexpected value must not silently disappear or crash --
    it passes through so a real fal 422 makes the gap obvious. (There's no
    translation table for quality any more, so this is really just
    confirming _translate_fal_params never touches the field.)"""
    result = worker._translate_fal_params({"quality": "ultra-mega"})
    assert result["quality"] == "ultra-mega"


# --------------------------------------------------------------------------- #
# size: our aspect-ratio strings -> fal's real image_size presets/object
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("our_value", [
    "original", "1:1", "9:16", "3:4", "2:3", "3:2", "4:3", "16:9",
])
def test_every_size_option_in_recolors_schema_translates_to_a_genuinely_valid_fal_image_size(our_value):
    """Cross-check against the literal size options /tools/recolor/schema
    serves (see options list in app/routes/tools.py's response for recolor)."""
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
    build_instruction() into the `prompt` string -- gpt-image-2/edit's real
    schema has no such fields (prompt, image_urls, image_size, background,
    quality, num_images, output_format, sync_mode, mask_url only)."""
    result = worker._translate_fal_params({"color": "red", "target_area": "sneaker sole", "quality": "low"})
    assert "color" not in result
    assert "target_area" not in result
    assert result["quality"] == "low"


def test_full_recolor_payload_translates_cleanly():
    params = {
        "color": "red",
        "quality": "medium",
        "size": "9:16",
        "target_area": "sneaker sole",
        "image_urls": ["https://fal.test/1.png"],
        "prompt": "Recolor the sneaker sole to color red",
    }

    result = worker._translate_fal_params(params)

    assert result == {
        "quality": "medium",
        "image_size": "portrait_16_9",
        "image_urls": ["https://fal.test/1.png"],
        "prompt": "Recolor the sneaker sole to color red",
    }


# --------------------------------------------------------------------------- #
# creative_photoshoot: its own "4:5" size option, "idea" stripped like
# color/target_area, and "quality" is a real 1K/2K/4K resolution tier
# translated to fal's actual quality enum + a default image_size.
# --------------------------------------------------------------------------- #

def test_creative_photoshoots_4_5_size_option_translates_to_a_genuinely_valid_fal_image_size():
    result = worker._translate_fal_params({"size": "4:5"})
    assert _is_valid_fal_image_size(result["image_size"]), (
        f"size='4:5' translated to {result['image_size']!r}, "
        f"which is not a value fal's image_size parameter actually accepts"
    )


def test_idea_is_stripped_not_forwarded_to_fal():
    """"idea" is creative_photoshoot's own UI/schema field, already consumed
    by build_instruction() into the `prompt` string -- gpt-image-2/edit's real
    schema has no such field."""
    result = worker._translate_fal_params({"idea": "Sci-Fi", "quality": "medium"})
    assert "idea" not in result
    assert result["quality"] == "medium"


@pytest.mark.parametrize("tier,expected_quality", [("1k", "medium"), ("2k", "high"), ("4k", "high")])
def test_creative_photoshoots_quality_tier_translates_to_a_real_fal_quality_value(tier, expected_quality):
    result = worker._translate_fal_params({"quality": tier})
    assert result["quality"] == expected_quality


@pytest.mark.parametrize("tier", ["1k", "2k", "4k"])
def test_creative_photoshoots_quality_tier_default_image_size_is_genuinely_valid(tier):
    """When no explicit aspect-ratio "size" is chosen, the quality tier's own
    resolution becomes the default image_size."""
    result = worker._translate_fal_params({"quality": tier})
    assert _is_valid_fal_image_size(result["image_size"]), (
        f"quality={tier!r} defaulted to image_size={result['image_size']!r}, "
        f"which is not a value fal's image_size parameter actually accepts"
    )


def test_explicit_aspect_ratio_size_wins_over_the_quality_tiers_default_resolution():
    """An explicit aspect-ratio choice ("9:16") controls dimensions; the
    quality tier still sets the real fal `quality` value, but does not
    override the size the user actually picked."""
    result = worker._translate_fal_params({"quality": "4k", "size": "9:16"})
    assert result["quality"] == "high"
    assert result["image_size"] == worker.FAL_SIZE_TRANSLATION["9:16"]


def test_quality_tier_still_supplies_its_default_resolution_when_size_is_original():
    result = worker._translate_fal_params({"quality": "2k", "size": "original"})
    assert result["quality"] == "high"
    assert result["image_size"] == worker.CREATIVE_QUALITY_TIERS["2k"]["image_size"]


def test_recolors_own_quality_values_are_unaffected_by_the_new_tier_table():
    """auto/low/medium/high (recolor's real schema values) must never match a
    key in CREATIVE_QUALITY_TIERS and must keep passing through unchanged."""
    for value in ("auto", "low", "medium", "high"):
        result = worker._translate_fal_params({"quality": value})
        assert result["quality"] == value
        assert "image_size" not in result  # no size given, no tier matched -- nothing to default


# --------------------------------------------------------------------------- #
# listing_photoshoot: "quality" (standard/high) translates to real, DISTINCT
# fal values -- selecting "high" must actually differ from "standard".
#
# Found live: this used to unconditionally force BOTH options to "medium"
# (the old FAL_HARDCODED_PARAMS blanket override) -- picking "high" quality
# silently did nothing at all. _translate_fal_params now needs feature_type
# explicitly passed for this to apply, since "quality":"high" is also
# recolor's own real, valid value and must not be touched for that tool.
# --------------------------------------------------------------------------- #

def test_listing_photoshoots_standard_quality_translates_to_medium():
    result = worker._translate_fal_params({"quality": "standard"}, feature_type="listing_photoshoot")
    assert result["quality"] == "medium"


def test_listing_photoshoots_high_quality_actually_differs_from_standard():
    """The real regression this fixes: "high" must not collapse to the same
    value as "standard" any more."""
    result = worker._translate_fal_params({"quality": "high"}, feature_type="listing_photoshoot")
    assert result["quality"] == "high"


def test_listing_photoshoots_quality_translation_only_applies_when_feature_type_matches():
    """Without feature_type="listing_photoshoot" (e.g. the default worker.py
    call for every other tool), "standard"/"high" must pass through
    untouched -- this translation must never apply globally."""
    result = worker._translate_fal_params({"quality": "standard"})
    assert result["quality"] == "standard"  # untouched -- no feature_type given


def test_listing_photoshoots_translation_does_not_leak_into_recolors_real_high_value():
    """recolor's own real "high" value must be completely unaffected even
    though it's spelled identically to listing_photoshoot's "high" option --
    proven by passing recolor's feature_type explicitly."""
    result = worker._translate_fal_params({"quality": "high"}, feature_type="recolor")
    assert result["quality"] == "high"  # unaffected -- LISTING_QUALITY_TRANSLATION never applies to recolor
