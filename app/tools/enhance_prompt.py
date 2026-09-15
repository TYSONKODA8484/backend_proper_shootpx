import logging

from app.models.tool_definition import ToolDefinition

logger = logging.getLogger(__name__)


def build_instruction(job, tool_definition, db) -> str:
    user_prompt = job.input_params.get("prompt")
    source_feature_type = job.input_params.get("source_feature_type")

    if not user_prompt:
        raise ValueError(f"job {job.id}: enhance_prompt called with no prompt to enhance")

    source_tool = db.query(ToolDefinition).filter(
        ToolDefinition.feature_type == source_feature_type
    ).first()
    if not source_tool:
        raise ValueError(f"job {job.id}: unknown source_feature_type {source_feature_type!r}")

    hint = source_tool.ai_steps.get("enhance_hint", {}).get("prompt_template")
    if not hint:
        logger.warning(
            "job %s: source tool %r has no ai_steps.enhance_hint configured -- "
            "enhancing with no tool-specific guidance", job.id, source_feature_type,
        )
        hint = "Rewrite the user's prompt to be clearer and more descriptive."

    logger.info("job %s: enhancing prompt for source tool %r", job.id, source_feature_type)
    return f"{hint}\n\nUser's prompt: {user_prompt}"