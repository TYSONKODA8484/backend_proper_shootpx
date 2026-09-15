"""app/tools/enhance_prompt.py::build_instruction -- the fallback hint used
when the source tool has no ai_steps.enhance_hint configured must come from
enhance_prompt's OWN tool_definitions row (ai_steps.default_hint), not a
hardcoded Python string, so it's DB-configurable without a deploy."""

from unittest.mock import MagicMock

from app.tools import enhance_prompt


def _job(prompt="make it better", source_feature_type="recolor"):
    return MagicMock(input_params={"prompt": prompt, "source_feature_type": source_feature_type})


def _db_with_source_tool(source_tool):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = source_tool
    return db


def test_uses_source_tools_enhance_hint_when_configured():
    source_tool = MagicMock(ai_steps={"enhance_hint": {"prompt_template": "Recolor-specific guidance."}})
    db = _db_with_source_tool(source_tool)
    tool_definition = MagicMock(ai_steps={"default_hint": "should not be used"})

    instruction = enhance_prompt.build_instruction(_job(), tool_definition, db)

    assert instruction == "Recolor-specific guidance.\n\nUser's prompt: make it better"


def test_falls_back_to_its_own_tool_definitions_default_hint_when_source_has_none():
    source_tool = MagicMock(ai_steps={})  # no enhance_hint configured
    db = _db_with_source_tool(source_tool)
    tool_definition = MagicMock(ai_steps={"default_hint": "A DB-configured generic hint."})

    instruction = enhance_prompt.build_instruction(_job(), tool_definition, db)

    assert instruction == "A DB-configured generic hint.\n\nUser's prompt: make it better"


def test_falls_back_to_the_literal_default_when_even_its_own_default_hint_is_missing():
    """Last-resort default if enhance_prompt's own row is ever misconfigured
    -- must never crash or produce an empty hint."""
    source_tool = MagicMock(ai_steps={})
    db = _db_with_source_tool(source_tool)
    tool_definition = MagicMock(ai_steps={})  # no default_hint either

    instruction = enhance_prompt.build_instruction(_job(), tool_definition, db)

    assert instruction == "Rewrite the user's prompt to be clearer and more descriptive.\n\nUser's prompt: make it better"


def test_raises_when_source_feature_type_is_unknown():
    db = _db_with_source_tool(None)

    try:
        enhance_prompt.build_instruction(_job(source_feature_type="ghost_tool"), MagicMock(ai_steps={}), db)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "ghost_tool" in str(e)


def test_raises_when_prompt_is_missing():
    db = _db_with_source_tool(MagicMock(ai_steps={}))

    try:
        enhance_prompt.build_instruction(_job(prompt=None), MagicMock(ai_steps={}), db)
        assert False, "expected ValueError"
    except ValueError:
        pass
