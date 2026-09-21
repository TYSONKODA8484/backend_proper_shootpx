import logging

from app.core.fal_client import call_fal_sync
from app.tools.prompts import require_prompt

logger = logging.getLogger(__name__)

DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_RESOLUTION = "standard"


def build_instruction(job, tool_definition, db) -> str:
    """
    Returns the final instruction string to send to the real recolor model.
    Two paths, per Recolor's PRD:
      - target_area blank -> call the vision model, its answer IS the
        finished instruction (the color is already baked into its prompt)
      - target_area given -> build the instruction directly, no AI call
    Unchanged by the flux-2/edit swap -- instruction text doesn't depend
    on which generation model actually receives it.
    """
    color = job.input_params.get("color")
    target_area = job.input_params.get("target_area")
    ai_steps = tool_definition.ai_steps

    if target_area:
        logger.info("job %s: target_area given (%r) -- skipping the vision step", job.id, target_area)
        return require_prompt(
            ai_steps, "direct_prompt_template", feature_type=job.feature_type,
        ).format(target_area=target_area, color=color)

    fallback_instruction = require_prompt(
        ai_steps, "fallback_prompt_template", feature_type=job.feature_type,
    ).format(color=color)

    vision_step = ai_steps.get("detect_target")
    if not vision_step:
        logger.warning(
            "job %s: target_area blank but no ai_steps.detect_target configured "
            "for tool %s -- using the generic fallback instruction, moondream NOT called",
            job.id, job.feature_type,
        )
        return fallback_instruction

    prompt = require_prompt(
        ai_steps, "detect_target", "prompt_template", feature_type=job.feature_type,
    ).format(color=color)
    image_urls = job.input_params.get("image_urls", [])
    image_url = image_urls[0] if image_urls else None

    logger.info("job %s: target_area blank -- calling vision model %s", job.id, vision_step["model"])

    result = call_fal_sync(
        model_id=vision_step["model"],
        input_params={
            "task_type": vision_step["task_type"],
            "image_url": image_url,
            "prompt": prompt,  # moondream-next's real field is "prompt", not "query"
        },
    )

    # moondream-next's real output schema is {"output": "..."} -- not "answer"
    instruction = result.get("output") or fallback_instruction
    logger.info("job %s: vision model returned %r -- final instruction: %r", job.id, result.get("output"), instruction)
    return instruction


def build_image_size(job, tool_definition) -> dict:
    """
    Returns the {"width": ..., "height": ...} dict to send as flux-2/edit's
    `image_size` input -- looked up from ai_steps.size_map using the user's
    chosen aspect_ratio + resolution.

    Defensive on purpose, same philosophy as build_instruction(): a bad or
    missing frontend value should degrade to a sane default (1:1 / standard),
    never crash the job. Billing already reads credit_cost off the SAME
    param_schema options the frontend sent, so this never charges for a size
    it didn't actually generate.
    """
    ai_steps = tool_definition.ai_steps
    size_map = ai_steps.get("size_map")
    if not size_map:
        logger.warning(
            "job %s: no ai_steps.size_map configured for tool %s -- "
            "falling back to hardcoded 1024x1024",
            job.id, job.feature_type,
        )
        return {"width": 1024, "height": 1024}

    aspect_ratio = job.input_params.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    resolution = job.input_params.get("resolution", DEFAULT_RESOLUTION)

    ratio_entry = size_map.get(aspect_ratio)
    if not ratio_entry:
        logger.warning(
            "job %s: aspect_ratio %r not in size_map for tool %s -- falling back to %r",
            job.id, aspect_ratio, job.feature_type, DEFAULT_ASPECT_RATIO,
        )
        ratio_entry = size_map.get(DEFAULT_ASPECT_RATIO, {})

    size = ratio_entry.get(resolution)
    if not size:
        logger.warning(
            "job %s: resolution %r not in size_map[%r] for tool %s -- falling back to %r",
            job.id, resolution, aspect_ratio, job.feature_type, DEFAULT_RESOLUTION,
        )
        size = ratio_entry.get(DEFAULT_RESOLUTION, {"width": 1024, "height": 1024})

    logger.info("job %s: aspect_ratio=%r resolution=%r -> image_size=%r", job.id, aspect_ratio, resolution, size)
    return size