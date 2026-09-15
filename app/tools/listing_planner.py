import json
import logging

from app.core.fal_client import call_fal_sync

logger = logging.getLogger(__name__)

DEFAULT_SHOT_TYPES = ["hero", "side", "wearing", "sole"]

_DEFAULT_FALLBACK_SHOT_PROMPT_TEMPLATE = "Product photo, {shot_type} angle, studio lighting, white background"

_DEFAULT_SYSTEM_PROMPT = (
    "You are planning a product listing photoshoot. You will be shown one or "
    "more product photos. Respond with ONLY valid JSON, no markdown, no code "
    "fences, no extra text -- a JSON array of exactly the requested number of "
    "objects, each shaped as {\"shot_type\": \"<short label, e.g. hero, side, "
    "wearing, sole, top, detail>\", \"prompt\": \"<a single, concrete, visual "
    "instruction for an image-editing model to produce that exact shot from "
    "the reference product photo>\"}."
)


def plan_listing_shots(tool_definition, image_urls: list[str], user_prompt: str | None, output_count: int) -> list[dict]:
    """
    One upfront call that looks at the product photo(s) and decides which N
    distinct listing shots to produce, writing a distinct generation prompt
    for each. Returns a list of {"shot_type", "prompt"} dicts, length ==
    output_count. If the user gave a prompt, it's passed as a strong
    override hint, not ignored.
    """
    planner_step = tool_definition.ai_steps.get("shot_planner", {})
    model_id = planner_step.get("model_id", "openrouter/router/vision")
    model = planner_step.get("model", "google/gemini-2.5-flash")
    system_prompt = planner_step.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
    fallback_template = planner_step.get("fallback_shot_prompt_template", _DEFAULT_FALLBACK_SHOT_PROMPT_TEMPLATE)

    instruction = f"Plan exactly {output_count} distinct listing shots for this product."
    if user_prompt:
        instruction += f" The user specifically wants: {user_prompt}. Prioritize this above default shot choices."

    result = call_fal_sync(
        model_id=model_id,
        input_params={
            "image_urls": image_urls,
            "prompt": instruction,
            "system_prompt": system_prompt,
            "model": model,
        },
    )

    shots = _parse_shots(result.get("output", ""), output_count, fallback_template)
    logger.info("planned %d listing shots: %r", len(shots), [s["shot_type"] for s in shots])
    return shots


def _fallback_shots(output_count: int, template: str = _DEFAULT_FALLBACK_SHOT_PROMPT_TEMPLATE) -> list[dict]:
    types = (DEFAULT_SHOT_TYPES * ((output_count // len(DEFAULT_SHOT_TYPES)) + 1))[:output_count]
    return [
        {"shot_type": t, "prompt": template.format(shot_type=t)}
        for t in types
    ]


def _parse_shots(raw_output: str, output_count: int, fallback_template: str = _DEFAULT_FALLBACK_SHOT_PROMPT_TEMPLATE) -> list[dict]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        shots = json.loads(cleaned)
        if not isinstance(shots, list) or not shots:
            raise ValueError("not a non-empty JSON array")
        for shot in shots:
            if "shot_type" not in shot or "prompt" not in shot:
                raise ValueError("shot missing shot_type/prompt")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "listing planner returned unparseable JSON (%s) -- falling back "
            "to default shot types: %r", e, raw_output,
        )
        return _fallback_shots(output_count, fallback_template)

    if len(shots) > output_count:
        shots = shots[:output_count]
    elif len(shots) < output_count:
        shots.extend(_fallback_shots(output_count - len(shots), fallback_template))

    return shots