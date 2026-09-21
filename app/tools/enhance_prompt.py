import logging

from app.services.tool_definitions import get_tool_definition
from app.tools.prompts import require_prompt

logger = logging.getLogger(__name__)


def build_instruction(job, tool_definition, db) -> str:
    user_prompt = job.input_params.get("prompt")
    source_feature_type = job.input_params.get("source_feature_type")

    if not user_prompt:
        raise ValueError(f"job {job.id}: enhance_prompt called with no prompt to enhance")

    source_tool = get_tool_definition(db, source_feature_type)
    if not source_tool:
        raise ValueError(f"job {job.id}: unknown source_feature_type {source_feature_type!r}")

    hint = source_tool.ai_steps.get("enhance_hint", {}).get("prompt_template")
    if not hint:
        logger.warning(
            "job %s: source tool %r has no ai_steps.enhance_hint configured -- "
            "enhancing with the generic default hint instead", job.id, source_feature_type,
        )
        # enhance_prompt's OWN row (tool_definition here, not source_tool).
        hint = require_prompt(
            tool_definition.ai_steps, "default_hint", feature_type=job.feature_type,
        )

    logger.info("job %s: enhancing prompt for source tool %r", job.id, source_feature_type)
    return require_prompt(
        tool_definition.ai_steps, "instruction_template", feature_type=job.feature_type,
    ).format(hint=hint, user_prompt=user_prompt)