from PIL import Image
import os
from pathlib import Path
import torch
import diffusers
import cv2
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from config import MODEL_ZOO_DIR


def draw_boxes(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4)
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)

        # print(ann)
        name = ann['name'] if 'name' in ann else str(ann['category_id'])
        # ax.text(bbox_x, bbox_y, name, style='italic',
        ax.text(bbox_x+20, bbox_y+25, name, style='normal',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
    
    # color[-1] = np.zeros((1,3))

    p = PatchCollection(polygons, facecolor='none',
                        edgecolors=color, linewidths=2)
    ax.add_collection(p)


def show_boxes(gen_boxes, ind=None, show=False, save=False):
    if len(gen_boxes) == 0:
        return
    
    anns = [{'name': gen_box['name'], 'bbox': gen_box['bounding_box']} for gen_box in gen_boxes]

    box_scale = (512, 512)
    size = box_scale
    size_h, size_w = size

    # White background (to allow line to show on the edge)
    # I = np.ones((size[0]+4, size[1]+4, 3), dtype=np.uint8) * 255
    I = np.ones((size[0], size[1], 3), dtype=np.uint8) * 240
    # I = np.zeros((size[0], size[1], 3), dtype=np.uint8) + 
    plt.figure(figsize=(6,6))
    plt.tight_layout()
    plt.imshow(I)
    plt.axis('off')
    draw_boxes(anns)
    if show:
        plt.show()
    else:
        # print("Saved boxes visualizations to", f"{img_dir}/boxes.png", f"ind: {ind}")
        plt.savefig(str(Path(__file__).resolve().parent / "inputs" / "boxes.png"), bbox_inches='tight', pad_inches=0)


def main_generate(args):
    result_path = None
    if args["tool"] == "text_to_image_SDXL":
        import torch
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
        sdxl_path = str(MODEL_ZOO_DIR / "stable-diffusion-xl-base-1.0")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            sdxl_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        ).to("cuda")
        # for i in range(10):
        g = torch.Generator('cuda').manual_seed(17)
        image = pipe(args["input"]["text"], height=1024, width=1024, generator=g).images[0]
        image.save(args["output"])
        result_path = args["output"]
    elif args["tool"] == 'image_to_image_SD2':
        import torch
        from diffusers import AutoPipelineForImage2Image
        from diffusers.utils import make_image_grid, load_image

        sd21_path = str(MODEL_ZOO_DIR / "stable-diffusion-2-1-base")
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            sd21_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
        pipeline.enable_model_cpu_offload()
        # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
        pipeline.enable_xformers_memory_efficient_attention()

        # prepare image
        url = args["input"]['image']
        init_image = load_image(url)

        prompt = args["input"]['text']
        image = pipeline(prompt, image=init_image).images[0]
        image.save(args["output"])
        result_path = args["output"]
    elif args["tool"] == "layout_to_image_LMD":
        # if args["input"]["layout"] == "TBG":
        #     args["input"]["layout"] = args_layout
        # bb_draw = [{'name': args_layout[i][0], 'bounding_box': args_layout[i][1]} for i in range(len(args_layout))]
        # show_boxes(bb_draw)
        # os.sys.path.append('./LLM-groundedDiffusion')
        # import models
        # from models import sam
        # models.sd_key = "stable-diffusion-2-1-base"
        # models.sd_version = "sdv2"
        # models.model_dict = models.load_sd(
        #     key=models.sd_key,
        #     use_fp16=False,
        #     scheduler_cls=None
        # )
        # sam_model_dict = sam.load_sam()
        # models.model_dict.update(sam_model_dict)
        # import generation.lmd as generation
        # spec = {'prompt': args["input"]["text"], 'gen_boxes': args["input"]["layout"], 'bg_prompt': 'A realistic scene', 'extra_neg_prompt': ''}
        # output = generation.run(
        #         spec=spec,
        #         bg_seed=27159,
        #         fg_seed_start=123483948,
        #         frozen_step_ratio=0.5
        #     )
        # output = Image.fromarray(output.image)
        # output.save(args["output"])

        # if args["input"]["layout"] == "TBG":
            # args["input"]["layout"] = args_layout
        # bb_draw = [{'name': args_layout[i][0], 'bounding_box': args_layout[i][1]} for i in range(len(args_layout))]
        # show_boxes(bb_draw)
        os.sys.path.append('./LLM-groundedDiffusion')
        import models
        from models import sam
        models.sd_key = str(MODEL_ZOO_DIR / "diffusers-generation-text-box")
        models.sd_version = "sdv1.4"
        models.model_dict = models.load_sd(
            key=models.sd_key,
            use_fp16=False,
            scheduler_cls=None
        )
        sam_model_dict = sam.load_sam()
        models.model_dict.update(sam_model_dict)
        import generation.lmd_plus as generation
        bg_prompt = args["input"].get('bg_prompt', 'A realistic scene')
        spec = {'prompt': args["input"]["text"], 'gen_boxes': args["input"]["layout"], 'bg_prompt': bg_prompt, 'extra_neg_prompt': ''}
        output = generation.run(
                spec=spec,
                bg_seed=27159,
                fg_seed_start=123483948,
                frozen_step_ratio=0.5
            )
        output = Image.fromarray(output.image)
        output.save(args["output"])
        result_path = args["output"]
    elif args["tool"] == "layout_to_image_BoxDiff":
        if args["input"]["layout"] == "TBG":
            args["input"]["layout"] = args_layout
        os.sys.path.append('./BoxDiff')
        from config import RunConfig
        import torch
        config_run = RunConfig()
        from pipeline.sd_pipeline_boxdiff import BoxDiffPipeline
        from utils import ptp_utils, vis_utils
        from utils.ptp_utils import AttentionStore
        from run_sd_boxdiff import run_on_prompt

        sd21_path = str(MODEL_ZOO_DIR / "stable-diffusion-2-1-base")
        stable = BoxDiffPipeline.from_pretrained(
            sd21_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        ).to("cuda")

        objs = [i[0] for i in args["input"]["layout"]]
        prompt = args["input"]["text"]
        bbox = [i[1] for i in args["input"]["layout"]]
        for k in range(len(bbox)):
            bbox[k] = [bbox[k][0], bbox[k][1], min(bbox[k][0]+bbox[k][2],512), min(bbox[k][1]+bbox[k][3], 512)]

        _SUB_BEGIN_POS = []
        _SUB_END_POS = []
        for k in range(len(objs)):
            ind = prompt.lower().find(objs[k].lower())
            ind = len(prompt.lower()[0:ind].split(' ') )
            _SUB_BEGIN_POS.append(ind)
            _SUB_END_POS.append(ind + len(objs[k].split(' ')) - 1)
        token_indices = _SUB_END_POS

        config_run.bbox = bbox

        g = torch.Generator('cuda').manual_seed(1)
        controller = AttentionStore()
        image = run_on_prompt(prompt=prompt,
                                model=stable,
                                controller=controller,
                                token_indices=token_indices,
                                seed=g,
                                config=config_run)
        image.save(args["output"])
        result_path = args["output"]
    elif args["tool"] == "superresolution_SDXL":
        from diffusers import StableDiffusionXLImg2ImgPipeline
        import torch
        torch.set_float32_matmul_precision("high")
        refiner_path = str(MODEL_ZOO_DIR / "stable-diffusion-xl-refiner-1.0")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )

        pipe = pipe.to("cuda")
        prompt = ''
        init_image = Image.open(args["input"]["image"])
        init_image = init_image.resize((1024, 1024), Image.LANCZOS)
        # for i in range(10):
        image = pipe(
            prompt,
            image=init_image,
            strength=0.3, #0.3,
            aesthetic_score=7., #7., #7.0,
            num_inference_steps=50,
        ).images
        image[0].save(args["output"])
        result_path = args["output"]
    import torch
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    return result_path


def run_generate_tool(command_dict: dict) -> str | None:
    """
    执行生成类工具（text_to_image_SDXL / image_to_image_SD2 / layout_to_image_* / superresolution_SDXL）。
    command_dict 格式与原 input.json 一致，需包含 "tool", "input", "output" 等。
    返回写入的图片路径，失败或无输出时返回 None。
    """
    return main_generate(command_dict)


if __name__ == "__main__":
    _cmd = {
        "tool": "text_to_image_SDXL",
        "input": {"text": "The glass picture frame and metallic stand display the wooden photo on the nightstand."},
        "output": "inputs/0.png",
    }
    out = run_generate_tool(_cmd)
    print("Output path:", out)

    