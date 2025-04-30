import os
import numpy as np
from PIL import Image
import argparse
import torch
from diffusers import DDIMScheduler, DiffusionPipeline, UNet2DConditionModel, AutoencoderKL

from utils.se_parallel_utils import register_se_forward  # share encoder


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
@torch.no_grad()
def encode_prompt(pipe, prompt):
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    captions = [prompt]
    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}


@torch.no_grad()
def inference(vae, unet, scheduler, encoded_embeds, generator, device, weight_dtype, register_store):
    input_shape = (1, 4, 64, 64)
    input_noise = torch.randn(input_shape, generator=generator, device=device, dtype=weight_dtype)

    prompt_embed = encoded_embeds["prompt_embeds"]
    prompt_embed = prompt_embed.to(device, weight_dtype)

    pred_original_sample = predict_original(unet, scheduler, input_noise, prompt_embed, register_store)
    pred_original_sample = pred_original_sample / vae.config.scaling_factor

    image = vae.decode(pred_original_sample.to(dtype=vae.dtype)).sample.float()
    image = (image[0].detach().cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return image

def predict_original(unet, scheduler, latents, prompt_embeds, register_store):
    noises_pred = []
    for i, t in enumerate(scheduler.timesteps):
        register_store['se_step'] = i
        noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds)['sample']

        if register_store['use_parallel']:
            break

        noises_pred += [noise_pred]

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    if register_store['use_parallel']:
        batch_size = latents.shape[0]
        for i, t in enumerate(register_store['timesteps']):
            curr_noise = noise_pred[i * batch_size: (i + 1) * batch_size]
            noises_pred += [curr_noise]
            latents = scheduler.step(curr_noise, t, latents).prev_sample

    return latents


def main(args):
    device, dtype = "cuda", torch.float16

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, scheduler=scheduler, torch_dtype=dtype)
    pipe = pipe.to(device)

    register_store = {'se_step': None, 'skip_feature': None, 'mid_feature': None, 'lora_scale': None,
                      'use_parallel': False, 'timesteps': [], 'bs': 1}
    register_store['use_parallel'] = args.use_parallel
    register_se_forward(pipe.unet, register_store)

    ## Loopfree need ddim scheduler
    scheduler.set_timesteps(args.num_inference_steps)
    if register_store['use_parallel']:
        register_store['timesteps'] = scheduler.timesteps

    with open(args.captions_file, 'r', encoding='utf-8') as f:
        prompts = f.readlines()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        generator = (torch.Generator(device=device).manual_seed(args.seed) if args.seed else None)
        encoded_embeds = encode_prompt(pipe, prompt)
        image = inference(pipe.vae, pipe.unet, scheduler, encoded_embeds, generator, device, dtype, register_store)
        image_array = np.asarray(image).astype(np.uint8)
        image = Image.fromarray(np.transpose(image_array, (1, 2, 0)))
        image.save(f"{output_dir}/{i:05}.png")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of an inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default="captions.txt",
        required=False,
        help="Path to the captions file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        required=False,
        help="Random seed used for inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Random seed used for inference.",
    )

    parser.add_argument('--use_parallel', action="store_true", help="if use parallel encoder share for training")
    parser.add_argument('--num_inference_steps', type=int, default=4, help="num_inference_steps for share encoder")

    args = parser.parse_args()
    main(args)
