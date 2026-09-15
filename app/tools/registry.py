from app.tools import recolor, enhance_prompt

TOOL_HANDLERS = {
    "recolor": recolor.build_instruction,
    "enhance_prompt": enhance_prompt.build_instruction,
}