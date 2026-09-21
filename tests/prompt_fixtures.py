"""Realistic ai_steps payloads for tests.

All prompt text now lives in tool_definitions.ai_steps (see
app/scripts/migrate_prompts_to_db.py) and the tools raise if a key is
missing -- so tests must build tool definitions that look like a real,
fully-migrated row rather than an empty MagicMock.

The prompt templates themselves are imported from the migration script, so
these fixtures can never drift from what's actually written to the database.
The extra keys below (model/model_id/system_prompt) are the ones that were
already on the rows before that migration.
"""

import copy

from app.scripts.migrate_prompts_to_db import PROMPT_KEYS

# Already present on the real rows; the migration doesn't write these.
PRE_EXISTING_KEYS = {
    "recolor": {
        "detect_target": {
            "model": "fal-ai/moondream-next",
            "task_type": "query",
            "prompt_template": "Find the main product and recolor it to {color}",
        },
        # fal-ai/flux-2/edit's real image_size field -- an explicit
        # {width, height} object, looked up by aspect_ratio + resolution.
        # Mirrors the real tool_definitions.ai_steps.size_map row exactly.
        "size_map": {
            "1:1": {"high": {"width": 1408, "height": 1408}, "standard": {"width": 1024, "height": 1024}},
            "3:4": {"high": {"width": 1248, "height": 1664}, "standard": {"width": 864, "height": 1152}},
            "4:3": {"high": {"width": 1664, "height": 1248}, "standard": {"width": 1152, "height": 864}},
            "16:9": {"high": {"width": 1920, "height": 1080}, "standard": {"width": 1360, "height": 768}},
            "9:16": {"high": {"width": 1080, "height": 1920}, "standard": {"width": 768, "height": 1360}},
        },
    },
    "creative_photoshoot": {
        "scene_vision": {
            "model": "google/gemini-2.5-flash",
            "model_id": "openrouter/router/vision",
            "prompt_template": (
                "Describe this product photo as a creative scene for the following idea: {idea}. "
                "Write it as a single, concrete, visual scene description suitable as an image generation prompt."
            ),
        },
        # Mirrors the real tool_definitions.ai_steps row exactly (openai/
        # gpt-image-2.5/flare/edit) -- resolve_generation_params reads these
        # to turn aspect_ratio+resolution+quality into image_size + credit_cost.
        "size_map": {
            "1:1": {"1k": {"width": 1024, "height": 1024}, "2k": {"width": 2048, "height": 2048}, "4k": {"width": 2880, "height": 2880}},
            "3:4": {"1k": {"width": 768, "height": 1024}, "2k": {"width": 1920, "height": 2560}, "4k": {"width": 2400, "height": 3200}},
            "4:3": {"1k": {"width": 1024, "height": 768}, "2k": {"width": 2560, "height": 1920}, "4k": {"width": 3200, "height": 2400}},
            "16:9": {"1k": {"width": 1920, "height": 1080}, "2k": {"width": 2560, "height": 1440}, "4k": {"width": 3840, "height": 2160}},
            "9:16": {"1k": {"width": 1080, "height": 1920}, "2k": {"width": 1440, "height": 2560}, "4k": {"width": 2160, "height": 3840}},
        },
        "quality_multiplier": {"low": 1, "medium": 2, "high": 5, "xhigh": 9, "max": 16},
        "resolution_multiplier": {"1k": 2, "2k": 4, "4k": 6},
    },
    "enhance_prompt": {
        "model": "google/gemini-2.5-flash",
        "default_hint": "Rewrite the user's prompt to be clearer and more descriptive.",
    },
    "listing_photoshoot": {
        "shot_planner": {
            "model": "google/gemini-2.5-flash",
            "model_id": "openrouter/router/vision",
            "system_prompt": "You are planning a product listing photoshoot. Respond with ONLY valid JSON.",
            "fallback_shot_prompt_template": "Product photo, {shot_type} angle, studio lighting, white background",
        },
        # Mirrors the real tool_definitions.ai_steps row exactly (openai/
        # gpt-image-2.5/flare/edit) -- listing_planner.resolve_generation_params
        # reads these to turn aspect_ratio+resolution+quality+output_count
        # into image_size + total credit_cost (one priced unit per shot).
        "size_map": {
            "1:1": {"1k": {"width": 1024, "height": 1024}, "2k": {"width": 2048, "height": 2048}, "4k": {"width": 2880, "height": 2880}},
            "3:4": {"1k": {"width": 768, "height": 1024}, "2k": {"width": 1920, "height": 2560}, "4k": {"width": 2400, "height": 3200}},
            "4:3": {"1k": {"width": 1024, "height": 768}, "2k": {"width": 2560, "height": 1920}, "4k": {"width": 3200, "height": 2400}},
            "16:9": {"1k": {"width": 1920, "height": 1080}, "2k": {"width": 2560, "height": 1440}, "4k": {"width": 3840, "height": 2160}},
            "9:16": {"1k": {"width": 1080, "height": 1920}, "2k": {"width": 1440, "height": 2560}, "4k": {"width": 2160, "height": 3840}},
        },
        "quality_multiplier": {"low": 1, "medium": 2, "high": 5, "xhigh": 9, "max": 16},
        "resolution_multiplier": {"1k": 2, "2k": 4, "4k": 6},
    },
    "model_shoot": {
        "shoot_analyst": {
            "model": "google/gemini-2.5-flash",
            "model_id": "openrouter/router/vision",
            "system_prompt": "You are a fashion e-commerce shoot planner. Respond with ONLY valid JSON.",
        },
        # Mirrors the real tool_definitions.ai_steps row exactly (fal-ai/
        # bytedance/seedream/v4.5/edit) -- model_shoot.resolve_generation_params
        # reads these to turn aspect_ratio+resolution+output_count into
        # image_size + total credit_cost (one priced unit per pose).
        "size_map": {
            "1:1": {"1k": {"width": 1920, "height": 1920}, "2k": {"width": 3008, "height": 3008}, "4k": {"width": 4096, "height": 4096}},
            "3:4": {"1k": {"width": 1920, "height": 2560}, "2k": {"width": 2496, "height": 3328}, "4k": {"width": 3072, "height": 4096}},
            "4:3": {"1k": {"width": 2560, "height": 1920}, "2k": {"width": 3328, "height": 2496}, "4k": {"width": 4096, "height": 3072}},
            "16:9": {"1k": {"width": 3413, "height": 1920}, "2k": {"width": 3755, "height": 2112}, "4k": {"width": 4096, "height": 2304}},
            "9:16": {"1k": {"width": 1920, "height": 3413}, "2k": {"width": 2112, "height": 3755}, "4k": {"width": 2304, "height": 4096}},
        },
        "resolution_credit": {"1k": 4, "2k": 6, "4k": 8},
    },
    "model_shoot_generate_model": {
        "model_prompt_writer": {
            "model": "google/gemini-2.5-flash",
            "model_id": "openrouter/router",
            "system_prompt": "Write a studio model prompt from the given attributes.",
        },
    },
}


def _deep_merge(base: dict, extra: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def ai_steps_for(feature_type: str, **overrides) -> dict:
    """A fully-configured ai_steps dict for this tool, exactly as a migrated
    row looks. Pass overrides to simulate a partially-configured row."""
    steps = _deep_merge(PRE_EXISTING_KEYS.get(feature_type, {}), PROMPT_KEYS.get(feature_type, {}))
    return _deep_merge(steps, overrides) if overrides else steps
