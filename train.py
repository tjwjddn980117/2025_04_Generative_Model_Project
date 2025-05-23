import os
import random
from tqdm.auto import tqdm
import torch
from torch.cuda.amp import autocast
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler
)
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from torchvision import transforms

# ---------------------
# Dataset Definition
# ---------------------
default_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

class PairedImageDataset(Dataset):
    def __init__(self, rgb_paths, target_paths, transform=default_transform):
        assert len(rgb_paths) == len(target_paths), "Mismatched dataset lengths"
        self.rgb_paths = rgb_paths
        self.target_paths = target_paths
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return {
            "rgb": self.transform(rgb),
            "target": self.transform(tgt)
        }

# ---------------------
# Training Function
# ---------------------
def main():
    # Hyperparameters
    model_id = "timbrooks/instruct-pix2pix"
    rgb_dir = "./Our_data/rgb"
    ir_dir = "./Our_data/ir"
    batch_size = 4
    lr = 5e-6
    epochs = 100
    sample_size = 400
    output_dir = "./finetuned_model"

    # Device & accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator(mixed_precision="fp16")

    # BLIP for dynamic captioning
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # Prepare file lists and sample subset
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.png'))])
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    rgb_map = {os.path.splitext(f)[0]: os.path.join(rgb_dir, f) for f in rgb_files}
    ir_map = {os.path.splitext(f)[0]: os.path.join(ir_dir, f) for f in ir_files}
    keys = sorted(set(rgb_map) & set(ir_map))
    random.seed()
    keys = random.sample(keys, min(sample_size, len(keys)))

    rgb_paths = [rgb_map[k] for k in keys]
    target_paths = [ir_map[k] for k in keys]

    # DataLoader
    dataset = PairedImageDataset(rgb_paths, target_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load and prepare pipeline
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(device)
    pipeline.unet = pipeline.unet.half()
    pipeline.vae = pipeline.vae.half()
    if hasattr(pipeline, 'text_encoder'):
        pipeline.text_encoder = pipeline.text_encoder.half()
    pipeline.set_progress_bar_config(disable=True)
    pipeline = accelerator.prepare(pipeline)

    # Optimizer
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=lr)

    # Validation samples
    val_indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    val_samples = [dataset[i] for i in val_indices]
    to_pil = transforms.ToPILImage()

    # Training Loop
    for epoch in range(1, epochs+1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            rgb = batch['rgb'].to(device).half()
            target = batch['target'].to(device).half()

            # PIL for BLIP and pipeline
            rgb_pil = to_pil((rgb[0] * 0.5 + 0.5).clamp(0,1).cpu())
            blip_inputs = blip_processor(images=rgb_pil, return_tensors='pt').to(device)
            cap_ids = blip_model.generate(**blip_inputs, max_new_tokens=16)
            dynamic_caption = blip_processor.decode(cap_ids[0], skip_special_tokens=True)
            prompt = f"Convert this RGB image to an IR image. {dynamic_caption}"

                        # Forward pass: latent output for training
            with autocast():
                out = pipeline(
                    prompt=[prompt] * rgb.size(0),
                    image=[rgb_pil.resize((256,256))] * rgb.size(0),  # match num prompts
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    output_type='latent'
                )
                latents = out.images  # latents are returned in 'images' for output_type='latent'
                target_latents = (
                    pipeline.vae.encode(target)
                    .latent_dist.sample() * pipeline.vae.config.scaling_factor
                )
                loss = torch.nn.functional.mse_loss(latents, target_latents)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Checkpoint save
        ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        pipeline.save_pretrained(ckpt_dir)
        print(f"Saved checkpoint for epoch {epoch} at {ckpt_dir}")

        # Validation
        for i, sample in enumerate(val_samples):
            with torch.no_grad(), autocast():
                rgb_val = sample['rgb'].unsqueeze(0).to(device).half()
                rgb_val_pil = to_pil((rgb_val[0] * 0.5 + 0.5).clamp(0,1).cpu())
                val_out = pipeline(
                    prompt=["Convert this RGB image to an IR image."],
                    image=rgb_val_pil.resize((256,256)),
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    output_type='pil'
                ).images[0]
            val_out.save(os.path.join(ckpt_dir, f"val_{i+1}.png"))
        print(f"Validation samples saved at {ckpt_dir}")

    # Final save
    pipeline.save_pretrained(output_dir)
    print(f"Training complete. Final model saved at {output_dir}")

if __name__ == '__main__':
    main()
