"""app/tools/model_shoot_generate_model.py::build_instruction -- writes a
text-to-image prompt for an AI model photo from structured attributes
(gender/age_bracket/ethnicity/skin_tone/body_type) + optional notes, via a
vision-less LLM call (openrouter/router). The one hard business rule
(never generate a sexualized-context underage model) is enforced in PYTHON
against the free-text `notes` field BEFORE any AI call is made -- never
trusted blind to the LLM's own judgment.
"""

from unittest.mock import MagicMock

import pytest

from app.tools import model_shoot_generate_model as msgm
from tests.prompt_fixtures import ai_steps_for


def _job(notes="", gender="Female", age_bracket="Adult (30s-40s)", ethnicity="South Asian",
         skin_tone="Fair", body_type="Slim"):
    return MagicMock(id="job-1", input_params={
        "notes": notes, "gender": gender, "age_bracket": age_bracket,
        "ethnicity": ethnicity, "skin_tone": skin_tone, "body_type": body_type,
    })


WRITER_STEP = ai_steps_for("model_shoot_generate_model")["model_prompt_writer"]


def _tool(ai_steps=None):
    """Defaults to a fully-migrated row -- the attribute/notes templates and
    the fallback instruction all come from ai_steps now, never from Python."""
    return MagicMock(ai_steps=ai_steps if ai_steps is not None else ai_steps_for("model_shoot_generate_model"))


# --------------------------------------------------------------------------- #
# minor-age notes blocking -- a range of real phrasings, not just "child"
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("notes", [
    "make her look like a child",
    "this is a minor",
    "teen model please",
    "looks like a kid",
    "toddler clothing shoot",
    "dressed as a baby",
    "underage appearance",
    "under 18 years old",
    "under-18 look",
    "under age model",
    "schoolgirl outfit",
    "school girl uniform",
    "schoolboy look",
    "school boy uniform",
    "she is 16 years old",
    "a 17-year-old model",
    "make the model 15 year old",
])
def test_blocks_a_range_of_real_minor_age_phrasings(notes):
    job = _job(notes=notes)
    tool = _tool()
    with pytest.raises(ValueError, match="Blocked"):
        msgm.build_instruction(job, tool, MagicMock())


@pytest.mark.parametrize("notes", [
    "",
    "adult model, confident pose",
    "she is 25 years old",
    "a 30-year-old model, mature look",
    "baby blue background",
    "baby pink dress",
    "this is childish humor in the ad copy",  # "child" must not match inside "childish"
    "kidney-shaped pool backdrop",            # "kid" must not match inside "kidney"
    "a mature adult, 45 year old",
])
def test_does_not_block_legitimate_adult_notes(notes, monkeypatch):
    monkeypatch.setattr(msgm, "call_fal_sync", MagicMock(return_value={"output": "A professional studio portrait."}))
    job = _job(notes=notes)
    tool = _tool()

    instruction = msgm.build_instruction(job, tool, MagicMock())

    assert instruction == "A professional studio portrait."


def test_blocked_notes_never_reach_the_ai_call(monkeypatch):
    call_fal_sync = MagicMock()
    monkeypatch.setattr(msgm, "call_fal_sync", call_fal_sync)
    job = _job(notes="she should look like a 12 year old")
    tool = _tool()

    with pytest.raises(ValueError):
        msgm.build_instruction(job, tool, MagicMock())

    call_fal_sync.assert_not_called()


def test_age_number_at_or_above_18_is_not_blocked():
    msgm._check_notes_for_minor_age("she is 18 years old")  # must not raise
    msgm._check_notes_for_minor_age("a 21 year old model")  # must not raise


# --------------------------------------------------------------------------- #
# happy path -- structured attributes + notes -> AI-written prompt
# --------------------------------------------------------------------------- #

def test_writes_prompt_from_structured_attributes(monkeypatch):
    call_fal_sync = MagicMock(return_value={"output": "A professional studio portrait of an adult South Asian woman."})
    monkeypatch.setattr(msgm, "call_fal_sync", call_fal_sync)
    job = _job(notes="")
    tool = _tool()

    instruction = msgm.build_instruction(job, tool, MagicMock())

    assert instruction == "A professional studio portrait of an adult South Asian woman."
    call_fal_sync.assert_called_once_with(
        model_id="openrouter/router",
        input_params={
            "prompt": (
                "Structured attributes: {'gender': 'Female', 'age_bracket': 'Adult (30s-40s)', "
                "'ethnicity': 'South Asian', 'skin_tone': 'Fair', 'body_type': 'Slim'}"
            ),
            "system_prompt": WRITER_STEP["system_prompt"],  # straight from the DB row, not from code
            "model": "google/gemini-2.5-flash",
        },
    )


def test_notes_are_appended_to_the_prompt_when_present(monkeypatch):
    call_fal_sync = MagicMock(return_value={"output": "x"})
    monkeypatch.setattr(msgm, "call_fal_sync", call_fal_sync)
    job = _job(notes="warm smile, soft lighting")
    tool = _tool()

    msgm.build_instruction(job, tool, MagicMock())

    sent_prompt = call_fal_sync.call_args.kwargs["input_params"]["prompt"]
    assert sent_prompt.endswith("\nAdditional notes: warm smile, soft lighting")


def test_falls_back_to_a_generic_instruction_when_the_ai_gives_no_answer(monkeypatch):
    monkeypatch.setattr(msgm, "call_fal_sync", MagicMock(return_value={}))
    job = _job(notes="")
    tool = _tool()

    instruction = msgm.build_instruction(job, tool, MagicMock())

    assert instruction == "A professional studio portrait photo of an adult model, plain background"


def test_model_routing_comes_from_the_rows_ai_steps(monkeypatch):
    call_fal_sync = MagicMock(return_value={"output": "x"})
    monkeypatch.setattr(msgm, "call_fal_sync", call_fal_sync)
    job = _job(notes="")
    tool = _tool()

    msgm.build_instruction(job, tool, MagicMock())

    kwargs = call_fal_sync.call_args.kwargs
    assert kwargs["model_id"] == "openrouter/router"
    assert kwargs["input_params"]["model"] == "google/gemini-2.5-flash"


def test_missing_prompt_key_raises_naming_the_db_key(monkeypatch):
    """No prompt text is hidden in Python any more -- a row missing the
    writer's instruction_template must fail loudly, naming the exact key."""
    from app.tools.prompts import MissingPromptError

    call_fal_sync = MagicMock()
    monkeypatch.setattr(msgm, "call_fal_sync", call_fal_sync)
    steps = ai_steps_for("model_shoot_generate_model")
    del steps["model_prompt_writer"]["instruction_template"]

    try:
        msgm.build_instruction(_job(notes=""), _tool(ai_steps=steps), MagicMock())
        assert False, "expected MissingPromptError"
    except MissingPromptError as e:
        assert "ai_steps.model_prompt_writer.instruction_template" in str(e)

    call_fal_sync.assert_not_called()
