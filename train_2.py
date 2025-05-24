import os
import torch
from torch.cuda.amp import autocast
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler
)
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from torchvision import transforms
import numpy as np

# Transform helpers

def default_image_transform(size=(256,256)):
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


def default_mask_transform(size=(256,256)):
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.NEAREST),
        transforms.ToTensor(),  # yields 1-channel mask
    ])

# 1. Paired Dataset 정의 (RGB, IR, Segmentation Mask)
class PairedImageDataset(Dataset):
    def __init__(
        self,
        rgb_paths,
        target_paths,
        mask_paths,
        image_transform=None,
        mask_transform=None
    ):
        self.rgb_paths = rgb_paths
        self.target_paths = target_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform or default_image_transform()
        self.mask_transform = mask_transform or default_mask_transform()

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_pil = Image.open(self.rgb_paths[idx]).convert("RGB")
        tgt_pil = Image.open(self.target_paths[idx]).convert("RGB")
        mask_pil = Image.open(self.mask_paths[idx]).convert("L")

        rgb = self.image_transform(rgb_pil)
        tgt = self.image_transform(tgt_pil)
        mask = self.mask_transform(mask_pil)

        return {"rgb": rgb, "target": tgt, "mask": mask}

# 2. 학습 함수
def main():
    model_id = "timbrooks/instruct-pix2pix"
    controlnet_id = "lllyasviel/sd-controlnet-seg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    lr = 5e-5
    epochs = 10
    output_dir = "./finetuned_model"

    accelerator = Accelerator(mixed_precision="fp16")

    # BLIP for captioning
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # Prepare dataset paths
    rgb_dir = "./Our_data/rgb"
    ir_dir = "./Our_data/ir"
    seg_dir = "./Our_data/seg"
    rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg')]
    ir_files = [f for f in os.listdir(ir_dir) if f.lower().endswith('.jpeg')]
    seg_files = [f for f in os.listdir(seg_dir) if f.lower().endswith('.png')]

    rgb_map = {os.path.splitext(f)[0]: os.path.join(rgb_dir, f) for f in rgb_files}
    ir_map = {os.path.splitext(f)[0]: os.path.join(ir_dir, f) for f in ir_files}
    seg_map = {os.path.splitext(f)[0]: os.path.join(seg_dir, f) for f in seg_files}
    common_keys = sorted(set(rgb_map) & set(ir_map) & set(seg_map))
    rgb_paths = [rgb_map[k] for k in common_keys]
    target_paths = [ir_map[k] for k in common_keys]
    mask_paths = [seg_map[k] for k in common_keys]

    dataset = PairedImageDataset(rgb_paths, target_paths, mask_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ControlNet and Pipeline
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.float16
    ).to(device)

    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    # Cast sub-models to fp16
    pipeline = pipeline.to(device)
    pipeline.unet = pipeline.unet.half()
    pipeline.controlnet = pipeline.controlnet.half()
    pipeline.vae = pipeline.vae.half()
    if hasattr(pipeline, 'text_encoder'):
        pipeline.text_encoder = pipeline.text_encoder.half()

    pipeline = accelerator.prepare(pipeline)

    optimizer = torch.optim.AdamW(
        list(pipeline.unet.parameters()) + list(pipeline.controlnet.parameters()),
        lr=lr
    )

    to_pil = transforms.ToPILImage()
    for epoch in range(epochs):
        for batch in dataloader:
            # Move and cast to fp16
            rgb = batch["rgb"].to(device).half()
            target = batch["target"].to(device).half()
            mask = batch["mask"].to(device).half()

            # PIL conversion for BLIP & ControlNet
            rgb_pil = to_pil((rgb[0] * 0.5 + 0.5).clamp(0, 1).cpu())
            mask_pil = to_pil(mask[0].cpu())

            # Generate caption
            blip_inputs = blip_processor(images=rgb_pil, return_tensors="pt").to(device)
            caps = blip_model.generate(**blip_inputs, max_new_tokens=16)
            prompt = blip_processor.decode(caps[0], skip_special_tokens=True)
            prompt = prompt + 'Turn to IR image.'

            # Encode and noise under autocast
            with autocast():
                target_latent = (
                    pipeline.vae.encode(target)
                    .latent_dist.sample() * pipeline.vae.config.scaling_factor
                )
                noise = torch.randn_like(target_latent)
                timesteps = torch.randint(
                    0, pipeline.scheduler.config.num_train_timesteps,
                    (target_latent.shape[0],), device=device
                )
                noisy_latent = pipeline.scheduler.add_noise(target_latent, noise, timesteps)

                # Text embeddings
                text_inputs = pipeline.tokenizer(
                    [prompt] * rgb.size(0), padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    return_tensors="pt"
                ).to(device)
                text_embeds = pipeline.text_encoder(**text_inputs).last_hidden_state

                # Prepare mask conditioning tensor
                mask_cond = mask.repeat(1, 3, 1, 1)

                # ControlNet forward
                controlnet_output = pipeline.controlnet(
                    noisy_latent,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=mask_cond
                )
                ctrl_out = controlnet_output[0]

                # UNet forward with cross_attention_kwargs
                unet_out = pipeline.unet(
                    noisy_latent,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    cross_attention_kwargs={"controlnet_cond": ctrl_out}
                ).sample

            # Compute loss and step
            loss = torch.nn.functional.mse_loss(unet_out, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    pipeline.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
