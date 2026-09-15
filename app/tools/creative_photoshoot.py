import logging

from app.core.fal_client import call_fal_sync

logger = logging.getLogger(__name__)


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
        return f"{idea}. {prompt}" if prompt else idea

    image_urls = job.input_params.get("image_urls", [])

    query = vision_step["prompt_template"].format(idea=idea)
    if prompt:
        query += f" The user also specifically asked for: {prompt}"

    result = call_fal_sync(
        model_id=vision_step["model_id"],
        input_params={
            "image_urls": image_urls,
            "prompt": query,
            "model": vision_step["model"],
        },
    )

    instruction = result.get("output") or (f"{idea}. {prompt}" if prompt else idea)
    logger.info(
        "job %s: vision model returned %r -- final instruction: %r",
        job.id, result.get("output"), instruction,
    )
    return instruction