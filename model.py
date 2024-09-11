import os
from PIL import Image
import random
from io import BytesIO
import controlnet_aux

import diffusers
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler, LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.models import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel
)
import numpy as np
import requests
import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer
)
# const
tokenizer_path = 'botp/stable-diffusion-v1-5'
tokenizer_subfolder = 'tokenizer'
clip_path = 'botp/stable-diffusion-v1-5'
clip_subfolder = 'text_encoder'
unet_path = 'botp/stable-diffusion-v1-5'
unet_subfolder = 'unet'
controlnet_path = 'lllyasviel/sd-controlnet-depth'
vae_path = 'botp/stable-diffusion-v1-5'
vae_subfolder = 'vae'

# args
# device = "cuda:7"
# guidance_scale = 7.5
# vae_scaling_factor = 1

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def save_image(images, image_path_dir, image_name_prefix):
    """
    Save the generated images to png files.
    """
    images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    for i in range(images.shape[0]):
        image_path  = os.path.join(image_path_dir, image_name_prefix+str(i+1)+'-'+str(random.randint(1000,9999))+'.png')
        print(f"Saving image {i+1} / {images.shape[0]} to: {image_path}")
        Image.fromarray(images[i]).save(image_path)

class Controlnet_pipeline():
    def __init__(
            self,
            height=512,
            width=512,
            unet_channels=4,
            guidance_scale=7.5,
            vae_scaling_factor=0.18215,
            device_id=0,
            device="cuda:0",
            dtype = torch.float16,
            seed=777
    ):
        print("[I] Pipeline start initiating")
        self.height = height
        self.width = width
        self.unet_channels = unet_channels
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = (guidance_scale > 1.0)
        self.vae_scaling_factor = vae_scaling_factor
        self.device_id = device_id
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.basic_models = {
            "tokenizer": CLIPTokenizer.from_pretrained(tokenizer_path, subfolder=tokenizer_subfolder, use_safetensors=False),
            "clip": CLIPTextModel.from_pretrained(clip_path, subfolder=clip_subfolder, use_safetensors=True).to(device),
            "controlnet": ControlNetModel.from_pretrained(controlnet_path, torch_dtype=self.dtype).to(device),
            "unet": UNet2DConditionModel.from_pretrained(unet_path, subfolder=unet_subfolder, use_safetensors=True, torch_dtype=self.dtype).to(device),
            "vae": AutoencoderKL.from_pretrained(vae_path, subfolder=vae_subfolder, use_safetensors=True).to(device),
            "scheduler": UniPCMultistepScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler")
        }
        print("[I] Finish loading models from cache")
        self.latent_height = self.height // 8
        self.latent_width = self.width // 8
        self.base_path = "controlnet_test_result"

        self.engines = {}
        self.shared_device_memory = None
        self.contexts = {}
        self.tensors = {}

    def infer_torch(
            self,
            prompt,
            negative_prompt,
            input_images,
            denoising_steps, 
            controlnet_scales,
            output_path="output"
    ):
        with torch.inference_mode(), torch.autocast("cuda"):
            self.basic_models['scheduler'].set_timesteps(denoising_steps, device=self.device)
            timesteps = self.basic_models['scheduler'].timesteps.to(self.device)
            latents = self.basic_models['scheduler'].init_noise_sigma * torch.randn(
                (len(prompt), self.unet_channels, self.latent_height, self.latent_width), device=self.device, dtype=torch.float32, generator=self.generator)

            text_input_ids = self.basic_models["tokenizer"](
                prompt, 
                padding="max_length", 
                max_length=self.basic_models["tokenizer"].model_max_length, 
                truncation=True, 
                return_tensors="pt",
            ).input_ids.type(torch.int32).to(self.device)
            text_embeddings = self.basic_models["clip"](text_input_ids, output_hidden_states=False)[0]

            if self.do_classifier_free_guidance:
                negative_text_input_ids = self.basic_models["tokenizer"](
                    negative_prompt, 
                    padding="max_length", 
                    max_length=self.basic_models["tokenizer"].model_max_length, 
                    truncation=True, 
                    return_tensors="pt",
                ).input_ids.type(torch.int32).to(self.device)
                uncond_embeddings = self.basic_models["clip"](negative_text_input_ids, output_hidden_states=False)[0]

                text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)


            for _, timestep in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.basic_models['scheduler'].scale_model_input(latent_model_input, timestep)

                # down_samples, mid_sample = self.basic_models['controlnet'](
                #     latent_model_input,
                #     timestep,
                #     encoder_hidden_states=text_embeddings,
                #     controlnet_cond=input_images,
                #     conditioning_scale=controlnet_scales,
                #     return_dict=False,
                # )

                noise_pred = self.basic_models['unet'](
                    latent_model_input, 
                    timestep, 
                    encoder_hidden_states=text_embeddings,
                    # down_block_additional_residuals=down_samples,
                    # mid_block_additional_residual=mid_sample
                )['sample']

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.basic_models["scheduler"].step(noise_pred, timestep, latents, return_dict=False)[0]

            latents = 1. / self.vae_scaling_factor * latents

            images = self.basic_models['vae'].decode(latents)['sample']
       
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            save_image(images, "output", "controlnet_test_{}_".format(self.seed))
       
# # init
# height = 512
# width = 512
# unet_channels = 4
# guidance_scale = 7.5
# vae_scaling_factor = 0.18215
# controlnet_scales = 1.0
# device_id = 7
# device = f"cuda:{device_id}"
# seed = 777

# # inference
# batch_size = 1
# num_warmup_runs = 4
# use_cuda_graph = False
# denoising_steps = 20

# prompt = ["Stormtrooper's lecture in beautiful lecture hall"]
# negative_prompt = [""]

# depth_image = download_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
# print("[I] Finish downloading image ")


# demo = Controlnet_pipeline(
#     height,
#     width,
#     unet_channels,
#     guidance_scale,
#     vae_scaling_factor,
#     device_id,
#     device,
#     seed
# )

# input_images = []
# depth_image = controlnet_aux.LeresDetector.from_pretrained("lllyasviel/Annotators")(depth_image)
# input_images.append(depth_image.resize((height, width)))
# input_images = [(np.array(i.convert("RGB")).astype(np.float32) / 255.0)[..., None].transpose(3, 2, 0, 1).repeat(len(prompt), axis=0) for i in input_images]
# input_images = [torch.cat( [torch.from_numpy(i).to(device).float()] * (2 if demo.do_classifier_free_guidance else 1) ) for i in input_images]
# input_images = torch.cat([image[None, ...] for image in input_images], dim=0)[0]

# demo.infer_torch(prompt, negative_prompt, input_images, denoising_steps, torch.FloatTensor([controlnet_scales]).to(demo.device))