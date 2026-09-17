import logging
import re

from app.core.fal_client import call_fal_sync
from app.tools.prompts import require_prompt

logger = logging.getLogger(__name__)

MINOR_AGE_BLOCKED_PATTERNS = [
    "child", "minor", "teen", "kid", "toddler", "baby", "underage", "under 18", "under-18",
    "under age", "schoolgirl", "school girl", "schoolboy", "school boy",
]
MINOR_AGE_NUMBER_RE = re.compile(r"\b(\d{1,2})\s*[-\s]?year", re.IGNORECASE)


def _check_notes_for_minor_age(text: str) -> None:
    if not text:
        return
    lowered = text.lower()
    for phrase in MINOR_AGE_BLOCKED_PATTERNS:
        pattern = rf"\b{re.escape(phrase)}\b"
        if phrase == "baby":
            pattern += r"(?!\s+(?:blue|pink)\b)"
        if re.search(pattern, lowered):
            raise ValueError(f"Blocked: notes imply a minor/underage subject ('{phrase}').")
    for match in MINOR_AGE_NUMBER_RE.finditer(lowered):
        if int(match.group(1)) < 18:
            raise ValueError(f"Blocked: notes specify an age under 18 ('{match.group(0)}').")


def build_instruction(job, tool_definition, db) -> str:
    notes = job.input_params.get("notes", "")
    _check_notes_for_minor_age(notes)

    attrs = {
        "gender": job.input_params.get("gender"),
        "age_bracket": job.input_params.get("age_bracket"),
        "ethnicity": job.input_params.get("ethnicity"),
        "skin_tone": job.input_params.get("skin_tone"),
        "body_type": job.input_params.get("body_type"),
    }

    ai_steps = tool_definition.ai_steps
    feature_type = getattr(job, "feature_type", "model_shoot_generate_model")
    writer_step = ai_steps.get("model_prompt_writer", {})
    model_id = writer_step.get("model_id", "openrouter/router")
    model = writer_step.get("model", "google/gemini-2.5-flash")
    system_prompt = require_prompt(
        ai_steps, "model_prompt_writer", "system_prompt", feature_type=feature_type,
    )

    prompt_text = require_prompt(
        ai_steps, "model_prompt_writer", "instruction_template", feature_type=feature_type,
    ).format(attrs=attrs)
    if notes:
        prompt_text += require_prompt(
            ai_steps, "model_prompt_writer", "notes_suffix_template", feature_type=feature_type,
        ).format(notes=notes)

    result = call_fal_sync(
        model_id=model_id,
        input_params={"prompt": prompt_text, "system_prompt": system_prompt, "model": model},
    )

    instruction = result.get("output") or require_prompt(
        ai_steps, "model_prompt_writer", "fallback_instruction", feature_type=feature_type,
    )
    logger.info("job %s: model-generation prompt written: %r", job.id, instruction)
    return instruction