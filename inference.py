import argparse
import os
from omegaconf import OmegaConf
import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from diffusers import StableDiffusionInpaintPipeline
from dataset.dataset import TestDataset
from model.clip_away import CLIPAway
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/inference_config.yaml")
    return parser.parse_args()


def generate_focused_embeddings_grid(image, mask, fg_focused, bg_focused, projected, inpainted):
    images = [image, mask, fg_focused, bg_focused, projected, inpainted]
    row_image = Image.new('RGB', (image.width * len(images), image.height + 30))
    for i, img in enumerate(images):
        row_image.paste(img, (image.width * i, 30))

    draw = ImageDraw.Draw(row_image)
    font_path = os.path.join(os.path.dirname(__file__), "assets/OpenSans-Regular.ttf")
    font = ImageFont.truetype(font_path, 20)
    labels = ["Original Image", "Mask", "Unconditional Foreground Focused Generation",
            "Unconditional Background Focused Generation", "Unconditional CLIPAway Generation", 
            "Inpainted with CLIPAway"]

    for i, label in enumerate(labels):
        draw.text((image.width * i, 0), label, font=font, fill=(255, 255, 255))
    
    return row_image


def main(config):
    device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    print(f"Using device: {device}")
    
    sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        config.sd_model_key, safety_checker=None, torch_dtype=torch.float32
    )

    clipaway = CLIPAway(
        sd_pipe=sd_pipeline, 
        image_encoder_path=config.image_encoder_path,
        ip_ckpt=config.ip_adapter_ckpt_path, 
        alpha_clip_path=config.alpha_clip_ckpt_pth, 
        config=config, 
        alpha_clip_id=config.alpha_clip_id, 
        device=device, 
        num_tokens=4
    )

    test_dataset = TestDataset(config.root_path)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    
    # latent sizes are given according to StableDiffusionInpaintPipeline
    latents = torch.randn((1,4,64,64), generator=torch.Generator().manual_seed(config.seed)).to(device)

    for batch in test_dataloader:
        image, mask, image_paths = batch["image"], batch["mask"], batch["image_path"]
        image_pil = [ToPILImage()(img) for img in image]
        mask_pil = [ToPILImage()(img) for img in mask]

        final_image = clipaway.generate(
            prompt=[""], scale=config.scale, seed=config.seed,
            pil_image=image_pil, alpha=mask_pil, strength=config.strength, latents=latents
        )[0]
        
        if config.display_focused_embeds:
            full_mask = Image.new('L', (mask_pil[0].width, mask_pil[0].height), 255)
            projected_embeds, fg_embeds, bg_embeds, uncond_image_prompt_embeds = clipaway.get_focused_embeddings(
                image_pil, mask_pil, use_projection_block=True
            )
            
            fg_image = clipaway.generate(
                prompt=[""], image_prompt_embeds=fg_embeds, uncond_image_prompt_embeds=uncond_image_prompt_embeds,
                scale=config.scale, seed=config.seed, pil_image=image_pil,
                alpha=full_mask, strength=config.strength, latents=latents
            )[0]

            bg_image = clipaway.generate(
                prompt=[""], image_prompt_embeds=bg_embeds, uncond_image_prompt_embeds=uncond_image_prompt_embeds,
                scale=config.scale, seed=config.seed, pil_image=image_pil,
                alpha=full_mask, strength=config.strength, latents=latents
            )[0]

            proj_image = clipaway.generate(
                prompt=[""], image_prompt_embeds=projected_embeds, uncond_image_prompt_embeds=uncond_image_prompt_embeds,
                scale=config.scale, seed=config.seed, pil_image=image_pil,
                alpha=full_mask, strength=config.strength, latents=latents
            )[0]
            
            final_image = generate_focused_embeddings_grid(
                image_pil[0], mask_pil[0], fg_image, bg_image, proj_image, final_image
            )
        
        final_image.save(f"{config.save_path_prefix}/{os.path.basename(image_paths[0])}")


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    os.makedirs(config.save_path_prefix, exist_ok=True)
    main(config)
