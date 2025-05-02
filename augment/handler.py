import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline,ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from segment_anything import SamPredictor, sam_model_registry


class ModelHandler:
    def __init__(self, model_id, device):
        self.accelerator = Accelerator()
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
        ).to(device)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)

    def generate_images(self, prompt, img_path, num_images, guidance_scale):
        image = Image.open(img_path).convert('RGB').resize((256, 256))
        return self.pipeline(prompt, image=image, num_images_per_prompt=num_images, guidance_scale=guidance_scale).images

class ControlNetModelHandler:
    def __init__(self, controlnet_id, model_id, device):
        self.accelerator = Accelerator()
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
        ).to(device)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        # 'facebook/sam-vit-h' 리포지토리에서 자동으로 체크포인트 가져오기
        self.sam_ckpt = hf_hub_download(
            repo_id="scenario-labs/sam_vit",
            filename="sam_vit_h_4b8939.pth",    # 스크린샷에서 보신 파일명
            repo_type="model"
        )
        self.sam = sam_model_registry["vit_h"](checkpoint=self.sam_ckpt).to(device)
        #self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth").to(device)
        self.predictor = SamPredictor(self.sam)

    def generate_images(self, prompt, img_path, num_images, guidance_scale, strength = 0.5, controlnet_conditioning_scale = 0.7):
        image = Image.open(img_path).convert('RGB').resize((256, 256))
        image_np = np.array(image)  # (H, W, 3)
        self.predictor.set_image(image_np)
        masks, _, _ = self.predictor.predict(
            point_coords=None,             # 포인트 조건 없이
            point_labels=None,
            box=None,                      # 박스 없이 전체 이미지
            multimask_output=False
        )
        seg_mask = masks[0].astype(np.uint8) * 255  # (H, W), 0/255

        # PIL 흑백 이미지로 변환
        seg_map = Image.fromarray(seg_mask).convert("L")
        seg_map.show()  

        return self.pipeline(prompt, init_image=image, image=seg_map,
                             strength=strength, controlnet_conditioning_scale=controlnet_conditioning_scale, 
                             num_images_per_prompt=num_images, guidance_scale=guidance_scale,).images