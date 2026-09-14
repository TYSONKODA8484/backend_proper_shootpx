from app.core.fal_client import submit_to_fal
from app.core.fal_client import call_fal_sync


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
        return f"Recolor the {target_area} to color {color}"

    vision_step = tool_definition.ai_steps.get("detect_target")
    if not vision_step:
        # Defensive fallback -- should never happen if tool_definitions is
        # configured correctly, but never crash the job over a missing
        # config entry.
        return f"Recolor the main product to color {color}"

    prompt = vision_step["prompt_template"].format(color=color)
    image_urls = job.input_params.get("image_urls", [])
    image_url = image_urls[0] if image_urls else None

    result = call_fal_sync(
        model_id=vision_step["model"],
        input_params={
            "task_type": vision_step["task_type"],
            "image_url": image_url,
            "query": prompt,
        },
    )

    return result.get("answer") or f"Recolor the main product to color {color}"