from typing import List, Tuple, Union, Optional
from PIL import Image

import torch
import numpy as np
from typing import List

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.image_processor import VaeImageProcessor

class NoiseCutMixPipeline(StableDiffusionPipeline):

    def prepare_mask_latens(self, mask, batch_size, height, width, dtype, device):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        mask = self.mask_processor.preprocess(
            mask, height=height // self.vae_scale_factor, width=width // self.vae_scale_factor, resize_mode="default", crops_coords=None
        )
        mask = mask.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        return mask
    
    @torch.no_grad()
    def __call__(
        self,
        prompt_1: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        mask: Union[Image.Image, np.ndarray, torch.Tensor, List[Image.Image], List[np.ndarray], List[torch.Tensor]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 25,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True
    )-> Union[Tuple, StableDiffusionPipelineOutput]:
        
        assert len(prompt_1) == len(prompt_2) == len(mask)
    
        device = self._execution_device

        # prepare mask processor
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=False, do_convert_grayscale=True
        )
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt_1,
            height,
            width,
            None, # for callback_steps
            negative_prompt
        )

        self._guidance_scale = guidance_scale

        # No Need
        self._clip_skip = None
        self._cross_attention_kwargs = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt_1 is not None and isinstance(prompt_1, str):
            batch_size = 1
        elif prompt_1 is not None and isinstance(prompt_1, list):
            batch_size = len(prompt_1)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_1_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt_1,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        prompt_2_embeds, _ = self.encode_prompt(
            prompt_2,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_1_embeds, prompt_2_embeds])
        else:
            prompt_embeds = torch.cat([prompt_1_embeds, prompt_2_embeds])
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5.1 Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5.2 mask latent variables
        latents_mask =  self.prepare_mask_latens(
            mask,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device
        )

        # # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = None

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text_1, noise_pred_text_2 = noise_pred.chunk(3)
                    noise_pred_mix = latents_mask * noise_pred_text_1 + (1 - latents_mask) * noise_pred_text_2
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_mix - noise_pred_uncond)
                else:
                    noise_pred_text_1, noise_pred_text_2 = noise_pred.chunk(2)
                    noise_pred = latents_mask * noise_pred_text_1 + (1 - latents_mask) * noise_pred_text_2

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                progress_bar.update()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
