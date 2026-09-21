import json
import logging

from app.core.fal_client import call_fal_sync
from app.tools.prompts import require_prompt

logger = logging.getLogger(__name__)


def resolve_generation_params(job, tool_definition) -> dict:
    """Turns the user's aspect_ratio + resolution + output_count picks into
    what the fal call needs: exact width/height (from size_map), and the
    total credit cost (resolution_credit * output_count -- Seedream has no
    quality param, so resolution is a pure upsell tier, not a real cost
    pass-through). Raises if the combo isn't in size_map or output_count
    is out of range."""
    aspect_ratio = job.input_params.get("aspect_ratio", "1:1")
    resolution = job.input_params.get("resolution", "1k")
    output_count = job.input_params.get("output_count", tool_definition.default_output_count)

    if not isinstance(output_count, int) or output_count < 1 or output_count > 8:
        raise ValueError(
            f"job {job.id}: output_count={output_count!r} is out of the allowed "
            f"range (1-8)"
        )

    size_map = tool_definition.ai_steps.get("size_map", {})
    dimensions = size_map.get(aspect_ratio, {}).get(resolution)
    if not dimensions:
        raise ValueError(
            f"job {job.id}: aspect_ratio={aspect_ratio!r} does not support "
            f"resolution={resolution!r} -- not in size_map"
        )

    resolution_credit = tool_definition.ai_steps["resolution_credit"][resolution]
    credit_cost = resolution_credit * output_count

    logger.info(
        "job %s: aspect_ratio=%s resolution=%s output_count=%s -> "
        "dimensions=%s credit_cost=%s",
        job.id, aspect_ratio, resolution, output_count, dimensions, credit_cost,
    )

    return {
        "image_size": dimensions,
        "output_count": output_count,
        "credit_cost": credit_cost,
    }


def flatten_garment_image_urls(garments: list[dict]) -> list[str]:
    """Flattens grouped {"label": str, "image_urls": [...]} entries into one
    ordered URL list -- the SAME order plan_model_shoot() below uses when
    building its vision-call labels, so worker.py's real generation call
    sees images in the identical position the LLM's written prompts refer
    to by index (see _translate_fal_params's model_shoot branch)."""
    urls = []
    for garment in garments:
        urls.extend(garment.get("image_urls") or [])
    return urls


_BLOCKING_AGE_STATUSES = {"underage"}
_AGE_STATUSES = {"adult", "underage", "adult_uncertain"}
_GARMENT_CATEGORIES = {"intimate", "general"}


def plan_model_shoot(tool_definition, model_image_url: str, garments: list[dict],
                      reference_image_urls: list[str], user_prompt: str | None, output_count: int) -> dict:
    """
    One vision call does the classification (garment_category, model_age_status)
    and always writes output_count pose prompts regardless of that
    classification -- the blocking business rule itself (intimate garment +
    underage, or + ambiguous age when configured to) is a Python `if` below,
    never the LLM's own free-text "blocked" claim.
    Returns {"blocked": bool, "reason": str, "prompts": [str, ...]}.

    garments: list of {"label": str, "image_urls": list[str]} -- each entry is
    ONE physical product, possibly shown across multiple angle images. Images
    within a group get a "(i/n)" suffix so the LLM treats them as the SAME
    item, not separate products -- a flat image list gives the LLM no way to
    tell "3 angles of one top" apart from "a top + a bottom + a watch".
    """
    ai_steps = tool_definition.ai_steps
    feature_type = getattr(tool_definition, "feature_type", "model_shoot")
    step = ai_steps.get("shoot_analyst", {})
    model_id = step.get("model_id", "openrouter/router/vision")
    model = step.get("model", "google/gemini-2.5-flash")
    system_prompt = require_prompt(ai_steps, "shoot_analyst", "system_prompt", feature_type=feature_type)

    def label(key: str) -> str:
        return require_prompt(ai_steps, "shoot_analyst", key, feature_type=feature_type)

    labels = [label("model_image_label_template").format(index=1)]
    all_images = [model_image_url]

    garment_template = label("garment_image_label_template")
    for garment in garments:
        garment_label = garment.get("label") or "Garment"
        urls = garment.get("image_urls") or []
        total = len(urls)
        for i, url in enumerate(urls, start=1):
            part_suffix = f" ({i}/{total})" if total > 1 else ""
            # .format() ignores kwargs the template doesn't reference, so this
            # stays compatible with an older/stale template that has no
            # {label}/{part_suffix} placeholders at all.
            labels.append(garment_template.format(
                index=len(labels) + 1, label=garment_label, part_suffix=part_suffix,
            ))
            all_images.append(url)

    for url in reference_image_urls:
        labels.append(label("reference_image_label_template").format(index=len(labels) + 1))
        all_images.append(url)

    instruction = require_prompt(
        ai_steps, "shoot_analyst", "instruction_template", feature_type=feature_type,
    ).format(labels="\n".join(labels), output_count=output_count)
    if user_prompt:
        # The real production template also references {output_count} (its
        # shot-type-override rule: "If the user names fewer/more shot types
        # than {output_count}...") -- found live via a KeyError here, since
        # the older/stale template this used to be tested against had no
        # such placeholder and never needed it.
        instruction += require_prompt(
            ai_steps, "shoot_analyst", "user_prompt_suffix_template", feature_type=feature_type,
        ).format(user_prompt=user_prompt, output_count=output_count)

    result = call_fal_sync(
        model_id=model_id,
        input_params={
            "image_urls": all_images,
            "prompt": instruction,
            "system_prompt": system_prompt,
            "model": model,
        },
    )

    # Defaults to True (fail closed) if the DB row predates this flag.
    age_uncertain_blocks = bool(step.get("age_uncertain_blocks", True))

    parsed = _parse_plan(result.get("output", ""), output_count, age_uncertain_blocks)
    logger.info("model_shoot plan: blocked=%s prompts=%d", parsed["blocked"], len(parsed["prompts"]))
    return parsed


def _parse_plan(raw_output: str, output_count: int, age_uncertain_blocks: bool) -> dict:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        garment_category = data["garment_category"]
        model_age_status = data["model_age_status"]
        if garment_category not in _GARMENT_CATEGORIES:
            raise ValueError(f"unexpected garment_category: {garment_category!r}")
        if model_age_status not in _AGE_STATUSES:
            raise ValueError(f"unexpected model_age_status: {model_age_status!r}")
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
        logger.error("model_shoot planner returned unusable output (%s): %r", e, raw_output)
        return {"blocked": True, "reason": "Could not safely plan this shoot -- please try again.", "prompts": []}

    # The ONLY business decision that matters: an intimate/undergarment
    # combined with a model classified underage (or, when age_uncertain_blocks
    # is on, adult_uncertain too) is blocked. This is a plain Python `if`
    # against two enum-validated classification fields -- the LLM never gets
    # to just say "blocked": true/false and be believed.
    blocking_statuses = _BLOCKING_AGE_STATUSES | ({"adult_uncertain"} if age_uncertain_blocks else set())
    blocked = garment_category == "intimate" and model_age_status in blocking_statuses
    if blocked:
        return {
            "blocked": True,
            "reason": f"garment is intimate/undergarment apparel, and the model's age was classified as {model_age_status!r}.",
            "prompts": [],
        }

    prompts = data.get("prompts", [])
    if len(prompts) != output_count:
        logger.error(
            "model_shoot planner returned unusable output (expected %d prompts, got %d): %r",
            output_count, len(prompts), raw_output,
        )
        return {"blocked": True, "reason": "Could not safely plan this shoot -- please try again.", "prompts": []}

    return {"blocked": False, "reason": "", "prompts": prompts}