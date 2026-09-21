"""Every prompt string this backend sends to a model lives in the database
(tool_definitions.ai_steps), never in Python.

This module is the single accessor for that text. It deliberately has NO
default prompt values: a missing key raises, loudly naming the exact
feature_type and ai_steps path to add, instead of silently substituting
hidden prompt text that would then drift out of sync with the row ops
actually edit. See app/scripts/migrate_prompts_to_db.py for the writer side.
"""


class MissingPromptError(ValueError):
    """Raised when a tool_definitions row is missing prompt text the tool needs."""


def require_prompt(ai_steps: dict, *path: str, feature_type: str = "") -> str:
    """
    Fetch prompt text at ai_steps[path...]. Raises MissingPromptError naming
    the exact missing key, so a misconfigured row is obvious in the job's
    error message and logs rather than silently degrading output quality.
    """
    value = ai_steps or {}
    for key in path:
        if not isinstance(value, dict) or key not in value:
            dotted = ".".join(path)
            raise MissingPromptError(
                f"tool_definitions row for {feature_type or '<unknown tool>'} is missing "
                f"ai_steps.{dotted} -- prompt text lives in the database, not in code. "
                f"Run: python -m app.scripts.migrate_prompts_to_db"
            )
        value = value[key]

    if not isinstance(value, str) or not value.strip():
        dotted = ".".join(path)
        raise MissingPromptError(
            f"tool_definitions row for {feature_type or '<unknown tool>'} has an empty "
            f"ai_steps.{dotted} -- prompt text must be a non-empty string"
        )
    return value
