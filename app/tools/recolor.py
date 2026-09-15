import logging

from app.core.fal_client import submit_to_fal
from app.core.fal_client import call_fal_sync

logger = logging.getLogger(__name__)


def build_instruction(job, tool_definition) -> str:
    """
    Returns the final instruction string to send to the real recolor model.
    Two paths, per Recolor's PRD:
      - target_area blank -> call the vision model, its answer IS the
        finished instruction (the color is already baked into its prompt)
      - target_area given -> build the instruction directly, no AI call
    """
    color = job.input_params.get("color")
    target_area = job.input_params.get("target_area")

    if target_area:
        logger.info("job %s: target_area given (%r) -- skipping the vision step", job.id, target_area)
        return f"Recolor the {target_area} to color {color}"

    vision_step = tool_definition.ai_steps.get("detect_target")
    if not vision_step:
        # Defensive fallback -- should never happen if tool_definitions is
        # configured correctly, but never crash the job over a missing
        # config entry.
        logger.warning(
            "job %s: target_area blank but no ai_steps.detect_target configured "
            "for tool %s -- using the generic fallback instruction, moondream NOT called",
            job.id, job.feature_type,
        )
        return f"Recolor the main product to color {color}"

    prompt = vision_step["prompt_template"].format(color=color)
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
    instruction = result.get("output") or f"Recolor the main product to color {color}"
    logger.info("job %s: vision model returned %r -- final instruction: %r", job.id, result.get("output"), instruction)
    return instruction