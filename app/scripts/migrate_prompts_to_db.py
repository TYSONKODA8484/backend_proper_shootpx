"""Moves every remaining hardcoded prompt string out of app/tools/*.py and
into tool_definitions.ai_steps, so the database is the single source of truth
for all prompt text.

Idempotent: sets exactly the keys below, preserving every other key already
on each row (model/model_id/system_prompt/etc). Safe to re-run.

MUST be run against every environment's database (dev AND prod) before
deploying the matching code change -- the tools now read these keys from the
DB and raise a clear "missing ai_steps key" error instead of falling back to
hidden prompt text in Python.

    python -m app.scripts.migrate_prompts_to_db
"""

import json

from sqlalchemy import text

from app.core.database import SessionLocal

# feature_type -> the ai_steps keys this migration adds. Nested dicts are
# merged one level deep (so scene_vision's existing model/model_id survive).
PROMPT_KEYS = {
    "recolor": {
        # The real generation prompt when the user names a target area --
        # no AI call happens on this path at all, this string IS the prompt.
        "direct_prompt_template": "Recolor the {target_area} to color {color}",
        # Used when detect_target isn't configured, or the vision model
        # returns nothing usable.
        "fallback_prompt_template": "Recolor the main product to color {color}",
    },
    "creative_photoshoot": {
        "scene_vision": {
            "user_prompt_suffix_template": " The user also specifically asked for: {prompt}",
        },
        "fallback_instruction_template": "{idea}. {prompt}",
        "fallback_instruction_idea_only_template": "{idea}",
    },
    "enhance_prompt": {
        "instruction_template": "{hint}\n\nUser's prompt: {user_prompt}",
    },
    "listing_photoshoot": {
        "shot_planner": {
            "instruction_template": "Plan exactly {output_count} distinct listing shots for this product.",
            "user_prompt_suffix_template": (
                " The user specifically wants: {user_prompt}. Prioritize this above default shot choices."
            ),
            "default_shot_types": ["hero", "side", "wearing", "sole"],
        },
    },
    "model_shoot": {
        "shoot_analyst": {
            "model_image_label_template": "#Image{index} = model reference",
            "garment_image_label_template": "#Image{index} = garment product, exact product, preserve fidelity",
            "reference_image_label_template": "#Image{index} = style/pose reference only",
            "instruction_template": (
                "Images in order:\n{labels}\n\n"
                "Write exactly {output_count} distinct production-ready image-edit prompts "
                "showing the garments on the model. Describe pose, framing, camera angle, lighting, "
                "and scene only -- never describe garment texture, stitching, color detail, or "
                "material, the image model already sees the real product and will render that itself."
            ),
            "user_prompt_suffix_template": "\n\nUser's additional direction: {user_prompt}",
        },
    },
    "model_shoot_generate_model": {
        "model_prompt_writer": {
            "instruction_template": "Structured attributes: {attrs}",
            "notes_suffix_template": "\nAdditional notes: {notes}",
            "fallback_instruction": "A professional studio portrait photo of an adult model, plain background",
        },
    },
}


def _merge(existing: dict, additions: dict) -> dict:
    merged = dict(existing or {})
    for key, value in additions.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def main() -> None:
    db = SessionLocal()
    try:
        for feature_type, additions in PROMPT_KEYS.items():
            row = db.execute(
                text("SELECT ai_steps FROM tool_definitions WHERE feature_type = :ft"),
                {"ft": feature_type},
            ).fetchone()

            if row is None:
                print(f"SKIP {feature_type}: no tool_definitions row in this database")
                continue

            current = row._mapping["ai_steps"] or {}
            merged = _merge(current, additions)

            if merged == current:
                print(f"OK   {feature_type}: already up to date")
                continue

            db.execute(
                text("UPDATE tool_definitions SET ai_steps = CAST(:ai_steps AS jsonb) WHERE feature_type = :ft"),
                {"ai_steps": json.dumps(merged), "ft": feature_type},
            )
            print(f"WROTE {feature_type}: added/updated {sorted(additions.keys())}")

        db.commit()
        print("\nDone. Now clear the tool-definition cache so the running app "
              "picks these up immediately: POST /admin/cache/clear")
    finally:
        db.close()


if __name__ == "__main__":
    main()
