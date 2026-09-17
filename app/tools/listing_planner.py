import json
import logging

from app.core.fal_client import call_fal_sync
from app.tools.prompts import MissingPromptError, require_prompt

logger = logging.getLogger(__name__)

# Hard cap on shots per listing_photoshoot batch -- mirrors the DB row's own
# param_schema.output_count.max (currently 8). Each shot is a distinct,
# separately-planned (billable vision call) and separately-billed generation,
# so this is enforced here (not just the generic services.generation.
# MAX_OUTPUT_COUNT=20 cap, which every tool shares) to keep a single request
# from planning -- and later trying to bill -- more shots than this tool is
# priced/tested for. Also enforced up front in the /generate route, before
# the billable shot-planning vision call ever runs (see app/routes/generation.py).
MIN_OUTPUT_COUNT = 1
MAX_OUTPUT_COUNT = 8


def resolve_generation_params(job, tool_definition) -> dict:
    """Turns the user's aspect_ratio + resolution + quality + output_count
    picks into what the fal calls actually need: exact width/height (from
    size_map), and the total credit cost to charge (quality_multiplier *
    resolution_multiplier * output_count -- one unit per shot). Raises if
    the size combo isn't in size_map, or if output_count is out of the
    allowed range."""
    aspect_ratio = job.input_params.get("aspect_ratio", "1:1")
    resolution = job.input_params.get("resolution", "1k")
    quality = job.input_params.get("quality", "medium")
    output_count = job.input_params.get("output_count", tool_definition.default_output_count)

    if not isinstance(output_count, int) or output_count < MIN_OUTPUT_COUNT or output_count > MAX_OUTPUT_COUNT:
        raise ValueError(
            f"job {job.id}: output_count={output_count!r} is out of the allowed "
            f"range ({MIN_OUTPUT_COUNT}-{MAX_OUTPUT_COUNT})"
        )

    size_map = tool_definition.ai_steps.get("size_map", {})
    dimensions = size_map.get(aspect_ratio, {}).get(resolution)
    if not dimensions:
        raise ValueError(
            f"job {job.id}: aspect_ratio={aspect_ratio!r} does not support "
            f"resolution={resolution!r} -- not in size_map"
        )

    quality_mult = tool_definition.ai_steps["quality_multiplier"][quality]
    resolution_mult = tool_definition.ai_steps["resolution_multiplier"][resolution]
    credit_cost = quality_mult * resolution_mult * output_count

    logger.info(
        "job %s: aspect_ratio=%s resolution=%s quality=%s output_count=%s -> "
        "dimensions=%s credit_cost=%s",
        job.id, aspect_ratio, resolution, quality, output_count, dimensions, credit_cost,
    )

    return {
        "image_size": dimensions,
        "quality": quality,
        "output_count": output_count,
        "credit_cost": credit_cost,
    }


def plan_listing_shots(tool_definition, image_urls: list[str], user_prompt: str | None, output_count: int) -> list[dict]:
    """
    One upfront call that looks at the product photo(s) and decides which N
    distinct listing shots to produce, writing a distinct generation prompt
    for each. Returns a list of {"shot_type", "prompt"} dicts, length ==
    output_count. If the user gave a prompt, it's passed as a strong
    override hint, not ignored.
    """
    ai_steps = tool_definition.ai_steps
    feature_type = getattr(tool_definition, "feature_type", "listing_photoshoot")
    planner_step = ai_steps.get("shot_planner", {})
    model_id = planner_step.get("model_id", "openrouter/router/vision")
    model = planner_step.get("model", "google/gemini-2.5-flash")
    system_prompt = require_prompt(ai_steps, "shot_planner", "system_prompt", feature_type=feature_type)
    fallback_template = require_prompt(
        ai_steps, "shot_planner", "fallback_shot_prompt_template", feature_type=feature_type,
    )
    shot_types = planner_step.get("default_shot_types") or []

    instruction = require_prompt(
        ai_steps, "shot_planner", "instruction_template", feature_type=feature_type,
    ).format(output_count=output_count)
    if user_prompt:
        instruction += require_prompt(
            ai_steps, "shot_planner", "user_prompt_suffix_template", feature_type=feature_type,
        ).format(user_prompt=user_prompt)

    result = call_fal_sync(
        model_id=model_id,
        input_params={
            "image_urls": image_urls,
            "prompt": instruction,
            "system_prompt": system_prompt,
            "model": model,
        },
    )

    shots = _parse_shots(result.get("output", ""), output_count, fallback_template, shot_types)
    logger.info("planned %d listing shots: %r", len(shots), [s["shot_type"] for s in shots])
    return shots


def _fallback_shots(output_count: int, template: str, shot_types: list[str]) -> list[dict]:
    if not shot_types:
        raise MissingPromptError(
            "tool_definitions row for listing_photoshoot is missing "
            "ai_steps.shot_planner.default_shot_types -- shot labels live in the "
            "database, not in code. Run: python -m app.scripts.migrate_prompts_to_db"
        )
    types = (shot_types * ((output_count // len(shot_types)) + 1))[:output_count]
    return [
        {"shot_type": t, "prompt": template.format(shot_type=t)}
        for t in types
    ]


def _parse_shots(raw_output: str, output_count: int, fallback_template: str, shot_types: list[str]) -> list[dict]:
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
        return _fallback_shots(output_count, fallback_template, shot_types)

    if len(shots) > output_count:
        shots = shots[:output_count]
    elif len(shots) < output_count:
        shots.extend(_fallback_shots(output_count - len(shots), fallback_template, shot_types))

    return shots