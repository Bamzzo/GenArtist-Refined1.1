from PIL import Image
import os
import os.path as osp
import torch
import diffusers
import cv2
import gc
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import base64

from config import OPENAI_API_KEY, WORK_DIR
from agent_tool_generate import run_generate_tool
from agent_tool_aux import run_aux_tool
from agent_tool_edit import run_edit_tool
from utils.llm_client import generate_reply, LLMClientError

def _safe_parse_llm_payload(raw: str) -> Any:
    """Parse LLM JSON/dict payload safely; no eval(). Returns dict/list or empty dict on failure."""
    import json
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip markdown code blocks if present
    for prefix, suffix in [("```json", "```"), ("```", "```")]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
        if raw.endswith(suffix):
            raw = raw[:-len(suffix)]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logging.getLogger(__name__).warning("_safe_parse_llm_payload: failed to parse, returning {}")
        return {}


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def command_parse(commands, text, text_bg, dir='inputs'):
    args = []
    generation_arg = None
    for i in range(len(commands)):
        command = commands[i]
        tool = command.get("tool", "")
        inp = command.get("input", command.get("instruction", ""))
        txt = command.get("text", command.get("instruction", ""))
        box = command.get("box")
        edit_val = command.get("edit", inp)

        if tool == 'edit':
            k = len(args)
            if box is not None:
                if command.get('intbox'):
                    bb = box
                    command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": inp, 'box': command.get('box', box)} }
            else:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": inp} }
            args.append(arg)

            arg = {"tool": "object_addition_anydoor",  "text": text, "text_bg": text_bg,
                    "output": osp.join(dir, str(k+2)+'.png'), "output_mask": osp.join(dir, str(k+2)+'_mask.png'),
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": edit_val, "layout": osp.join(dir, str(k+1)+'_mask.png') } }
            args.append(arg)

            arg = {"tool": "replace_anydoor", "output": osp.join(dir, str(k+3)+'.png'),
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": osp.join(dir, str(k+2)+'.png'),
                             "object_mask": osp.join(dir, str(k+2)+'_mask.png'), "mask": osp.join(dir, str(k+1)+'_mask.png'),  } }
            args.append(arg)
        elif tool == 'move':
            if command.get('intbox') and box is not None:
                bb = box
                command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
            k = len(args)
            arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": inp} }
            args.append(arg)

            arg = {"tool": "remove", "output":  osp.join(dir, str(k+2)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "mask": osp.join(dir, str(k+1)+'_mask.png')} }
            args.append(arg)

            arg = {"tool": "addition_anydoor", "output": osp.join(dir, str(k+3)+'.png'),
                "input": {"image": osp.join(dir, str(k+2)+'.png'), "object": osp.join(dir, str(k)+'.png'),
                        "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "mask": command.get("box", box)  } }
            args.append(arg)
        elif tool == 'addition':
            k = len(args)
            if command.get('intbox') and box is not None:
                bb = box
                command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
            arg = {"tool": "object_addition_anydoor",  "text": text, "text_bg": text_bg,
                    "output": osp.join(dir, str(k+1)+'.png'), "output_mask": osp.join(dir, str(k+1)+'_mask.png'),
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": inp, "layout": command.get("box", box) } }
            args.append(arg)

            arg = {"tool": "addition_anydoor", "output": osp.join(dir, str(k+2)+'.png'),
                "input": {"image": osp.join(dir, str(k)+'.png'), "object": osp.join(dir, str(k+1)+'.png'),
                        "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "mask": command.get("box", box)  } }
            args.append(arg)
        elif tool == 'remove':
            k = len(args)
            arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": inp, "mask_threshold": command.get("mask_thr", 0.0)} }
            if box is not None:
                if command.get('intbox'):
                    bb = box
                    command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
                arg['input']['box'] = command.get('box', box)
            args.append(arg)

            arg = {"tool": "remove", "output":  osp.join(dir, str(k+2)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "mask": osp.join(dir, str(k+1)+'_mask.png')} }
            args.append(arg)
        elif tool == 'instruction':
            k = len(args)
            arg = {"tool": "instruction", "output": osp.join(dir, str(k+1)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": txt} }
            args.append(arg)
        elif tool == 'edit_attribute':
            k = len(args)
            if box is not None:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": inp, 'box': box} }
            else:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": inp} }
            args.append(arg)

            arg = {"tool": "attribute_diffedit",
                    "output": osp.join(dir, str(k+2)+'.png'),
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": inp, "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "attr": txt } }
            args.append(arg)

        elif tool in ['text_to_image_SDXL', 'image_to_image_SD2', 'layout_to_image_LMD', 'layout_to_image_BoxDiff', 'superresolution_SDXL']:
            generation_arg = command
            generation_arg["output"] = osp.join(dir, '0.png')
    
    k = len(args)
    arg = {"tool": "superresolution_SDXL", "input": {"image": osp.join(dir, str(k)+'.png')}, "output": osp.join(dir, str(k+1)+'.png')}
    args.append(arg)
    if generation_arg is not None:
        args = [generation_arg] + args
    return args
@dataclass
class AgentState:
    """统一的 Agent 工作流状态。"""

    user_prompt: str
    bg_prompt: str = ""
    layout_data: List[Any] = field(default_factory=list)
    current_image: Optional[str] = None
    detection_results: Dict[str, Any] = field(default_factory=dict)
    correction_history: List[Any] = field(default_factory=list)
    error_msg: Optional[str] = None

    # 额外内部字段用于保存规划命令和完整命令序列
    planning_command: Optional[Dict[str, Any]] = None
    commands: List[Dict[str, Any]] = field(default_factory=list)


def node_planning(state: AgentState) -> AgentState:
    """调用 LLM 进行工具规划与布局生成，更新 bg_prompt / layout_data / planning_command。"""
    logger = logging.getLogger(__name__)
    logger.info("[Node: Planning] Generating initial layout and bg_prompt...")

    try:
        # 1) 工具规划（generation.txt）
        with open("prompts/generation.txt", "r", encoding="utf-8") as f:
            template = f.readlines()
        user_textprompt = f"Input: {state.user_prompt}"
        textprompt = f"{' '.join(template)} \n {user_textprompt}"

        logger.info("[Node: Planning] Requesting GPT-4o for generation planning...")
        gen_text_raw = generate_reply(system_prompt="", user_prompt=textprompt, image_b64=None, model="qwen-max")
        gen_text = _safe_parse_llm_payload(gen_text_raw)
        if not gen_text or "input" not in gen_text:
            raise ValueError("LLM returned invalid generation payload (missing 'input')")
        # 2) 布局与背景（bbox.txt）
        with open("prompts/bbox.txt", "r", encoding="utf-8") as f:
            template = f.readlines()
        user_textprompt = f"Caption: {state.user_prompt}"
        textprompt = f"{' '.join(template)} \n {user_textprompt}"
        logger.info("[Node: Planning] Requesting GPT-4o for bounding box layout...")
        bbox_text_raw = generate_reply(system_prompt="", user_prompt=textprompt, image_b64=None, model="qwen-max")
        bbox_text = _safe_parse_llm_payload(bbox_text_raw)
        if not bbox_text:
            bbox_text = {"layout": gen_text.get("input", {}).get("layout", []), "bg_prompt": state.bg_prompt or ""}

        gen_text.setdefault("input", {})["layout"] = bbox_text.get("layout", [])
        gen_text["input"]["bg_prompt"] = bbox_text.get("bg_prompt", "")

        state.bg_prompt = bbox_text.get("bg_prompt", "")
        state.layout_data = bbox_text.get("layout", [])
        state.planning_command = gen_text
        state.commands = [gen_text]

        logger.info("[Node: Planning] Planning completed.")
        state.error_msg = None
    except (LLMClientError, Exception) as e:
        logger.error("[Node: Planning] Failed: %s", e, exc_info=True)
        state.error_msg = f"planning_failed: {e}"
    return state


def node_generate(state: AgentState) -> AgentState:
    """根据规划命令生成初始图片，更新 current_image 和 commands。"""
    logger = logging.getLogger(__name__)
    logger.info("[Node: Generate] Generating initial image...")

    if not state.planning_command:
        msg = "planning_command is missing, cannot generate image."
        logger.error("[Node: Generate] %s", msg)
        state.error_msg = msg
        return state

    try:
        generation_command = [state.planning_command]
        seq_args = command_parse(generation_command, state.user_prompt, state.bg_prompt)
        # 约定第 0 个为生成节点
        gen_cmd = seq_args[0]
        output_path = run_generate_tool(gen_cmd)
        state.current_image = output_path
        state.commands = seq_args
        # 原子化落盘：生成完成后立即持久化到 outputs 目录
        if output_path and osp.isfile(output_path):
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            import shutil
            dest = WORK_DIR / "latest_generated.png"
            shutil.copy2(output_path, dest)
            logger.info("[Node: Generate] Image persisted to %s", dest)
        logger.info("[Node: Generate] Image generated at %s", output_path)
        state.error_msg = None
    except Exception as e:
        logger.error("[Node: Generate] Failed: %s", e, exc_info=True)
        state.error_msg = f"generate_failed: {e}"
    return state


def node_detect(state: AgentState) -> AgentState:
    """对当前图片做检测，更新 detection_results。异常时仅打 Warning，返回上一阶段已落盘图片，不中断流水线。"""
    logger = logging.getLogger(__name__)
    logger.info("[Node: Detect] Running detection on current image...")

    if not state.current_image:
        msg = "current_image is missing, cannot run detection."
        logger.error("[Node: Detect] %s", msg)
        state.error_msg = msg
        return state

    try:
        detect_cmd = {
            "tool": "detection",
            "input": {"image": state.current_image, "text": "TBG"},
        }
        result = run_aux_tool(detect_cmd) or {}
        state.detection_results = result
        logger.info("[Node: Detect] Detection finished with %d entries.", len(result.get("detection", [])))
        state.error_msg = None
    except Exception as e:
        logger.warning("[Node: Detect] Detection failed (bypass): %s. Keeping last image.", e, exc_info=True)
        state.detection_results = {}
        # 不设置 state.error_msg，强行返回上一阶段已落盘的图片，保证 API status=success
    return state


def node_correct(state: AgentState) -> AgentState:
    """调用多模态 LLM 进行校验与自纠错，必要时触发编辑工具。"""
    logger = logging.getLogger(__name__)
    logger.info("[Node: Correct] Verifying and self-correcting image...")

    if not state.current_image:
        msg = "current_image is missing, cannot run correction."
        logger.error("[Node: Correct] %s", msg)
        state.error_msg = msg
        return state

    try:
        with open("prompts/correction.txt", "r", encoding="utf-8") as f:
            template = f.readlines()

        user_textprompt = f"Caption: {state.user_prompt}\n"
        detection_list = state.detection_results.get("detection", [])
        user_textprompt += "I can give you the position of all objects in the image: " + str(detection_list) + "\n"
        user_textprompt += "You can use these as a reference for generating the bounding box position of objects"
        user_textprompt += "Please onlyoutput the editing operations through dict, do not output other analysis process. Return the result with only plain text, do not use any markdown or other style. All characters must be in English."

        textprompt = f"{' '.join(template)} \n {user_textprompt}"
        base64_image = encode_image(state.current_image)
        logger.info("[Node: Correct] Requesting GPT-4o for verification and correction...")
        correction_raw = generate_reply(
            system_prompt="",
            user_prompt=textprompt,
            image_b64=base64_image,
            model="qwen-vl-max",
        )
        correction_text = _safe_parse_llm_payload(correction_raw)
        if isinstance(correction_text, dict):
            state.correction_history.append(correction_text)
        elif isinstance(correction_text, list):
            state.correction_history.extend(correction_text)
        else:
            state.correction_history.append(correction_text)

        # 若无任何修改建议则直接返回
        if not correction_text:
            logger.info("[Node: Correct] No further corrections required.")
            state.error_msg = None
            return state

        # 组装命令并执行
        if isinstance(correction_text, list):
            commands = [state.planning_command] + correction_text if state.planning_command else correction_text
        else:
            commands = [state.planning_command, correction_text] if state.planning_command else [correction_text]

        seq_args = command_parse(commands, state.user_prompt, state.bg_prompt)
        state.commands = seq_args

        # 从第 1 个开始依次执行工具
        for i in range(1, len(seq_args)):
            cmd = seq_args[i]
            tool = cmd.get("tool")
            logger.info("[Node: Correct] Executing step %d with tool '%s'...", i, tool)
            if tool in ["object_addition_anydoor", "segmentation"]:
                _ = run_aux_tool(cmd)
            elif tool in ["addition_anydoor", "replace_anydoor", "remove", "instruction", "attribute_diffedit"]:
                out_path = run_edit_tool(cmd)
                if out_path:
                    state.current_image = out_path
            elif tool in ["text_to_image_SDXL", "image_to_image_SD2", "layout_to_image_LMD", "layout_to_image_BoxDiff", "superresolution_SDXL"]:
                out_path = run_generate_tool(cmd)
                if out_path:
                    state.current_image = out_path
            else:
                logger.warning("[Node: Correct] Unknown tool '%s', skipping.", tool)

        logger.info("[Node: Correct] Correction finished. Current image: %s", state.current_image)
        state.error_msg = None
    except (LLMClientError, Exception) as e:
        logger.warning("[Node: Correct] Correction failed (bypass): %s. Keeping last image.", e, exc_info=True)
        # 不设置 state.error_msg，强行返回上一阶段已落盘的图片，保证 API status=success
    return state


def run_agent_pipeline(prompt: str) -> dict:
    """
    执行完整的状态机流水线：planning -> generate -> detect -> correct（含自纠错循环）。
    供 Web 框架或 CLI 调用。

    Returns:
        - 成功: {"status": "success", "image_path": str, "history": list}
        - 失败: {"status": "error", "error_msg": str}
    """
    logger = logging.getLogger(__name__)
    state = AgentState(user_prompt=prompt)

    logger.info("=== GenArtist Agent Pipeline Start ===")

    # 规划
    state = node_planning(state)
    if state.error_msg:
        logger.error("Pipeline aborted at Planning: %s", state.error_msg)
        return {"status": "error", "error_msg": state.error_msg}

    # 生成
    state = node_generate(state)
    if state.error_msg:
        logger.error("Pipeline aborted at Generate: %s", state.error_msg)
        return {"status": "error", "error_msg": state.error_msg}

    # 检测
    state = node_detect(state)
    if state.error_msg:
        logger.error("Pipeline aborted at Detect: %s", state.error_msg)
        return {"status": "error", "error_msg": state.error_msg}

    # 自纠错主循环
    max_iterations = 3
    for it in range(max_iterations):
        logger.info("=== Correction iteration %d/%d ===", it + 1, max_iterations)
        prev_history_len = len(state.correction_history)
        state = node_correct(state)
        if state.error_msg:
            logger.error("Correction failed at iteration %d: %s", it + 1, state.error_msg)
            break
        if len(state.correction_history) == prev_history_len:
            logger.info("No new corrections; stopping correction loop.")
            break

    logger.info("=== GenArtist Agent Pipeline Finished ===")
    logger.info("Final image: %s", state.current_image)
    if state.error_msg:
        return {"status": "error", "error_msg": state.error_msg}
    return {
        "status": "success",
        "image_path": state.current_image,
        "history": state.correction_history,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    example_prompt = (
        "an oil painting, where a green vintage car, a black scooter on the left of it and a blue bicycle on "
        "the right of it, are parked near a curb, with three birds in the sky"
    )
    result = run_agent_pipeline(example_prompt)
    print("Result:", result)
