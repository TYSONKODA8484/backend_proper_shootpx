from app.tools import recolor, enhance_prompt, creative_photoshoot

TOOL_HANDLERS = {
    "recolor": recolor.build_instruction,
    "enhance_prompt": enhance_prompt.build_instruction,
    "creative_photoshoot": creative_photoshoot.build_instruction,
}