"""app/tools/enhance_prompt.py::build_instruction -- the fallback hint used
when the source tool has no ai_steps.enhance_hint configured must come from
enhance_prompt's OWN tool_definitions row (ai_steps.default_hint), not a
hardcoded Python string, so it's DB-configurable without a deploy."""

from unittest.mock import MagicMock

from app.tools import enhance_prompt
from tests.prompt_fixtures import ai_steps_for


def _job(prompt="make it better", source_feature_type="recolor"):
    return MagicMock(input_params={"prompt": prompt, "source_feature_type": source_feature_type})


def _db_with_source_tool(source_tool):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = source_tool
    return db


def test_uses_source_tools_enhance_hint_when_configured():
    source_tool = MagicMock(ai_steps={"enhance_hint": {"prompt_template": "Recolor-specific guidance."}})
    db = _db_with_source_tool(source_tool)
    tool_definition = MagicMock(ai_steps=ai_steps_for("enhance_prompt", default_hint="should not be used"))

    instruction = enhance_prompt.build_instruction(_job(), tool_definition, db)

    assert instruction == "Recolor-specific guidance.\n\nUser's prompt: make it better"


def test_falls_back_to_its_own_tool_definitions_default_hint_when_source_has_none():
    source_tool = MagicMock(ai_steps={})  # no enhance_hint configured
    db = _db_with_source_tool(source_tool)
    tool_definition = MagicMock(ai_steps=ai_steps_for("enhance_prompt", default_hint="A DB-configured generic hint."))

    instruction = enhance_prompt.build_instruction(_job(), tool_definition, db)

    assert instruction == "A DB-configured generic hint.\n\nUser's prompt: make it better"


def test_missing_default_hint_raises_naming_the_db_key_instead_of_using_hidden_code_text():
    """All prompt text lives in tool_definitions.ai_steps now -- when
    enhance_prompt's own row is misconfigured there is deliberately NO
    literal fallback in Python to silently paper over it. The error must name
    the exact missing key so the misconfiguration is obvious and fixable."""
    from app.tools.prompts import MissingPromptError

    source_tool = MagicMock(ai_steps={})  # no enhance_hint on the source tool
    db = _db_with_source_tool(source_tool)
    tool_definition = MagicMock(ai_steps={})  # and no default_hint either

    try:
        enhance_prompt.build_instruction(_job(), tool_definition, db)
        assert False, "expected MissingPromptError"
    except MissingPromptError as e:
        assert "ai_steps.default_hint" in str(e)


def test_raises_when_source_feature_type_is_unknown():
    db = _db_with_source_tool(None)

    try:
        enhance_prompt.build_instruction(
            _job(source_feature_type="ghost_tool"), MagicMock(ai_steps=ai_steps_for("enhance_prompt")), db,
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "ghost_tool" in str(e)


def test_raises_when_prompt_is_missing():
    db = _db_with_source_tool(MagicMock(ai_steps={}))

    try:
        enhance_prompt.build_instruction(_job(prompt=None), MagicMock(ai_steps=ai_steps_for("enhance_prompt")), db)
        assert False, "expected ValueError"
    except ValueError:
        pass
