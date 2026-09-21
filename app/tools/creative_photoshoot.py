import logging

from app.core.fal_client import call_fal_sync
from app.tools.prompts import require_prompt

logger = logging.getLogger(__name__)


def resolve_generation_params(job, tool_definition) -> dict:
    """Turns the user's aspect_ratio + resolution + quality picks into what
    the fal call actually needs: exact width/height (from size_map) and the
    credit cost to charge (quality_multiplier * resolution_multiplier).
    Raises if the combo isn't in size_map -- that's the enforcement point
    for "this combo hasn't been priced/tested yet"."""
    aspect_ratio = job.input_params.get("aspect_ratio", "1:1")
    resolution = job.input_params.get("resolution", "1k")
    quality = job.input_params.get("quality", "medium")

    size_map = tool_definition.ai_steps.get("size_map", {})
    dimensions = size_map.get(aspect_ratio, {}).get(resolution)
    if not dimensions:
        raise ValueError(
            f"job {job.id}: aspect_ratio={aspect_ratio!r} does not support "
            f"resolution={resolution!r} -- not in size_map"
        )

    quality_mult = tool_definition.ai_steps["quality_multiplier"][quality]
    resolution_mult = tool_definition.ai_steps["resolution_multiplier"][resolution]
    credit_cost = quality_mult * resolution_mult

    logger.info(
        "job %s: aspect_ratio=%s resolution=%s quality=%s -> "
        "dimensions=%s credit_cost=%s",
        job.id, aspect_ratio, resolution, quality, dimensions, credit_cost,
    )

    return {
        "image_size": dimensions,
        "quality": quality,
        "credit_cost": credit_cost,
    }


def _fallback_instruction(tool_definition, feature_type: str, idea: str, prompt: str | None) -> str:
    """Used whenever the vision step can't run or returns nothing -- the
    composition template itself lives in the DB, same as every other prompt."""
    if prompt:
        return require_prompt(
            tool_definition.ai_steps, "fallback_instruction_template", feature_type=feature_type,
        ).format(idea=idea, prompt=prompt)
    return require_prompt(
        tool_definition.ai_steps, "fallback_instruction_idea_only_template", feature_type=feature_type,
    ).format(idea=idea)


def build_instruction(job, tool_definition, db) -> str:
    idea = job.input_params.get("idea")
    prompt = job.input_params.get("prompt")

    if not idea and not prompt:
        raise ValueError(f"job {job.id}: creative_photoshoot needs at least an idea or a prompt")

    if idea:
        logger.info("job %s: idea=%r prompt=%r -- calling vision model", job.id, idea, prompt)
        return _call_vision(job, tool_definition, idea=idea, prompt=prompt)

    logger.info("job %s: prompt only -- skipping vision model", job.id)
    return prompt


def _call_vision(job, tool_definition, idea: str, prompt: str | None) -> str:
    vision_step = tool_definition.ai_steps.get("scene_vision")
    if not vision_step:
        logger.warning(
            "job %s: idea given but no ai_steps.scene_vision configured for tool %s "
            "-- falling back to idea/prompt text as-is, vision model NOT called",
            job.id, job.feature_type,
        )
        return _fallback_instruction(tool_definition, job.feature_type, idea, prompt)

    image_urls = job.input_params.get("image_urls", [])

    query = require_prompt(
        tool_definition.ai_steps, "scene_vision", "prompt_template", feature_type=job.feature_type,
    ).format(idea=idea)
    if prompt:
        query += require_prompt(
            tool_definition.ai_steps, "scene_vision", "user_prompt_suffix_template",
            feature_type=job.feature_type,
        ).format(prompt=prompt)

    result = call_fal_sync(
        model_id=vision_step["model_id"],
        input_params={
            "image_urls": image_urls,
            "prompt": query,
            "model": vision_step["model"],
        },
    )

    instruction = result.get("output") or _fallback_instruction(
        tool_definition, job.feature_type, idea, prompt,
    )
    logger.info(
        "job %s: vision model returned %r -- final instruction: %r",
        job.id, result.get("output"), instruction,
    )
    return instruction