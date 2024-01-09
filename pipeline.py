import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torchvision import transforms

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from motion import create_motion_field_and_warp_latents
from cross_attn import  CrossFrameAttnProcessor2_0
from diffusers.utils import BaseOutput

# from gpt import get_motion
from sam_test import get_mask
from falcon import get_label_and_motion

import matplotlib.pyplot as plt
from kornia.morphology import dilation

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

class TextToVideoPipelineOutput(BaseOutput): #ke thua BaseOutput
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        images (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`[List[bool]]`):
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


class Pipeline(StableDiffusionPipeline): #ke thua Stable diffusionPipeline
    r"""
    Pipeline for zero-shot text-to-video generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A [`UNet3DConditionModel`] to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`CLIPImageProcessor`]):
            A [`CLIPImageProcessor`] to extract features from generated images; used as inputs to the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker
        )
        processor = CrossFrameAttnProcessor2_0(batch_size=2)
        self.unet.set_attn_processor(processor)
        
    def forward_loop(self, x_t0, t0, t1, generator):
        """
        Perform DDPM forward process from time t0 to t1. This is the same as adding noise with corresponding variance.

        Args:
            x_t0:
                Latent code at time t0.
            t0:
                Timestep at t0.
            t1:
                Timestamp at t1.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.

        Returns:
            x_t1:
                Forward process applied to x_t0 from time t0 to t1.
        """
        eps = torch.randn(x_t0.size(), generator=generator, dtype=x_t0.dtype, device=x_t0.device)
        alpha_vec = torch.prod(self.scheduler.alphas[t0:t1])
        x_t1 = torch.sqrt(alpha_vec) * x_t0 + torch.sqrt(1 - alpha_vec) * eps
        return x_t1

    def backward_loop( #DDPM backward process
        self,
        latents,
        timesteps,
        prompt_embeds,
        guidance_scale,
        callback,
        callback_steps,
        num_warmup_steps,
        extra_step_kwargs,
        cross_attention_kwargs=None,
    ):
        """
        Perform backward process given list of time steps.

        Args:
            latents:
                Latents at time timesteps[0].
            timesteps:
                Time steps along which to perform backward process.
            prompt_embeds:
                Pre-generated text embeddings.
            guidance_scale:
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            extra_step_kwargs:
                Extra_step_kwargs.
            cross_attention_kwargs:
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            num_warmup_steps:
                number of warmup steps.

        Returns:
            latents:
                Latents of backward process output at time timesteps[-1].
        """
        do_classifier_free_guidance = guidance_scale > 1.0
        num_steps = (len(timesteps) - num_warmup_steps) // self.scheduler.order
        with self.progress_bar(total=num_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # add to timestep counter
                #self.timestep_counter.append(t)

                # expand the latents if we are doing classifier free guidance #latentx2 to use the cFG function with weight
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance #cfg
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1 #subtract the noise from latent->new latent but less noise
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        return latents.clone().detach()
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int] = 8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50, 
        guidance_scale: float = 7.5, 
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        motion_field_strength_x: float = 12, 
        motion_field_strength_y: float = 12, 
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        t0: int = 44, 
        t1: int = 47, 
        frame_ids: Optional[List[int]] = None,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            video_length (`int`, *optional*, defaults to 8):
                The number of generated video frames.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in video generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"numpy"`):
                The output format of the generated video. Choose between `"latent"` and `"numpy"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoPipelineOutput`] instead of
                a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            motion_field_strength_x (`float`, *optional*, defaults to 12):
                Strength of motion in generated video along x-axis. See the [paper](https://arxiv.org/abs/2303.13439),
                Sect. 3.3.1.
            motion_field_strength_y (`float`, *optional*, defaults to 12):
                Strength of motion in generated video along y-axis. See the [paper](https://arxiv.org/abs/2303.13439),
                Sect. 3.3.1.
            t0 (`int`, *optional*, defaults to 44):
                Timestep t0. Should be in the range [0, num_inference_steps - 1]. See the
                [paper](https://arxiv.org/abs/2303.13439), Sect. 3.3.1.
            t1 (`int`, *optional*, defaults to 47):
                Timestep t0. Should be in the range [t0 + 1, num_inference_steps - 1]. See the
                [paper](https://arxiv.org/abs/2303.13439), Sect. 3.3.1.
            frame_ids (`List[int]`, *optional*):
                Indexes of the frames that are being generated. This is used when generating longer videos
                chunk-by-chunk.

        Returns:
            [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoPipelineOutput`]:
                The output contains a `ndarray` of the generated video, when `output_type` != `"latent"`, otherwise a
                latent code of generated videos and a list of `bool`s indicating whether the corresponding generated
                video contains "not-safe-for-work" (nsfw) content..
        """
        assert video_length > 0 #video_length = 8
        if frame_ids is None:
            frame_ids = list(range(video_length))
        assert len(frame_ids) == video_length

        assert num_videos_per_prompt == 1

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        print(prompt) #list
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor #[256, 256]
        width = width or self.unet.config.sample_size * self.vae_scale_factor #[256, 256]

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Perform the first backward process up to time T_1
        x_1_t1 = self.backward_loop(
            timesteps=timesteps[: -t1 - 1],
            prompt_embeds=prompt_embeds,
            latents=latents,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            extra_step_kwargs=extra_step_kwargs,
            num_warmup_steps=num_warmup_steps,
        )
        scheduler_copy = copy.deepcopy(self.scheduler)
        
        # self.timestep_counter = []
        # self.scheduler = scheduler_copy
        X_0=self.backward_loop(
            timesteps=timesteps[-t1 - 1 :],
            prompt_embeds=prompt_embeds,
            latents=x_1_t1,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            extra_step_kwargs=extra_step_kwargs,
            num_warmup_steps=num_warmup_steps,
        )
        
        print("test start")
        #print the type of X_0
        print(type(X_0)) #torch.tensor
        image = self.decode_latents(X_0)
        del X_0
        torch.cuda.empty_cache()
         
        #print the type of image
        print(type(image)) #numpy array
        # get_label = get_label(prompt)
        labels,Object_motion = get_label_and_motion(prompt[0])
        # Object_motion = get_motion(image) #llava #Blip
        
        mask = get_mask(image,prompt[0],labels) #mask.shape = (512,512)
        print(mask.shape, Object_motion)
        print("test end")
        
        #create a dictionary of motion with left, right, up, down as key and  each one will have (2,2) as value
        motion_field_dict = {
            'left': [20, 0], #object moving left, pixel motion field is left
            'right': [-20, 0], #move right
            'up': [0, 20], #move up
            'down': [0, -20], #move down
            'left_up': [-20, -20],
            'left_down': [-20, 20],
            'right_up': [20, -20],
            'right_down': [20, 20],
        }
        
        motion_field_strength_x, motion_field_strength_y = motion_field_dict.get(Object_motion, motion_field_dict['right_down'])
        #write me the code to apply resize the mask to h and w of x_1_t1 and apply dilation
        mask = torch.from_numpy(mask)[None, None].to(x_1_t1.device).to(x_1_t1.dtype)
        mask = transforms.Resize(size=(64, 64), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        #dilaion??
        kernel = torch.ones(5, 5, device=x_1_t1.device, dtype=x_1_t1.dtype)
        mask = dilation(mask, kernel)[0]
        #m_0 = m_0.to(x_1_t1.device) # dtype=torch.uint8
        m_0 = mask[None]
        m_0 = (m_0 > 0.5).to(x_1_t1.dtype)
        
        # tensor = m_0.cpu().squeeze()  # Remove the batch and channel dimensions
        # plt.figure(figsize=(20,20))
        # plt.imshow(tensor, cmap='cool')  # Plot the 64x64 image in grayscale
        # plt.savefig(f'm0.png')
        # self.scheduler = scheduler_copy
        
        # Perform the second backward process up to time T_0
        x_1_t0 = self.backward_loop(
            timesteps=timesteps[-t1 - 1 : -t0 - 1],
            prompt_embeds=prompt_embeds,
            latents=x_1_t1,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            extra_step_kwargs=extra_step_kwargs,
            num_warmup_steps=0,
        )

        # Propagate first frame latents at time T_0 to remaining frames
        #x_2k_t0 = x_1_t0.repeat(video_length-1, 1, 1, 1)
         
        # x_k_t0= x_1_t0.repeat(video_length, 1, 1, 1) #create 8 frames
        # m_k_t0= m_0.repeat(video_length, 1, 1, 1) #create 8 frames
        
        x_k_t0 = [x_1_t0 for _ in range(video_length)]
        m_k_t0 = [m_0 for _ in range(video_length)]
        #warp mask  
        for k in range(1, video_length):
            x_k_1_warp = create_motion_field_and_warp_latents(
                motion_field_strength_x=motion_field_strength_x,
                motion_field_strength_y=motion_field_strength_y,
                latents=x_k_t0[k-1]
            )
            # tensor = x_k_1_warp.cpu().squeeze()  # Remove the batch dimension
            # fig, axs = plt.subplots(1, 4, figsize=(80, 20))
            # for i in range(tensor.shape[0]):
            #     axs[i].imshow(tensor[i], cmap='cool')  # Plot each 64x64 image
            # plt.savefig(f'x_k_1_warp{k}.png')
            
            m_k_1 =m_k_t0[0] if k==1 else create_motion_field_and_warp_latents(
                motion_field_strength_x=motion_field_strength_x,
                motion_field_strength_y=motion_field_strength_y,
                latents=m_k_t0[k-2]
            )
            
            # tensor = m_k_1.cpu().squeeze()  # Remove the batch and channel dimensions
            # plt.figure(figsize=(20,20))
            # plt.imshow(tensor, cmap='cool')  # Plot the 64x64 image in grayscale
            # plt.savefig(f'm_k_1{k}.png')
            # plt.close()
            
            m_k_t0[k-1] = m_k_1 
            m_hat_k_1 = create_motion_field_and_warp_latents(
                motion_field_strength_x= -motion_field_strength_x,
                motion_field_strength_y= -motion_field_strength_y,
                latents=m_k_t0[k-1]
            )
            
            # tensor = m_hat_k_1.cpu().squeeze() # Remove the batch and channel dimensions
            # plt.figure(figsize=(20,20))
            # plt.imshow(tensor, cmap='cool')  # Plot the 64x64 image in grayscale
            # plt.savefig(f'm_hat_k_1{k}.png')
            
            m_tot_k_1=create_motion_field_and_warp_latents(
                motion_field_strength_x=motion_field_strength_x,
                motion_field_strength_y=motion_field_strength_y,
                latents= ((m_k_1 + m_hat_k_1) > 0.5).to(torch.float16)
            )
            
            # tensor = (((m_k_1 + m_hat_k_1) > 0.5).to(torch.float16)).cpu().squeeze()  # Remove the batch and channel dimensions
            # plt.figure(figsize=(5, 5))
            # plt.imshow(tensor, cmap='cool')  # Plot the 64x64 image in grayscale
            # plt.savefig(f'm_k_1+m_hat-k_1{k}.png')
            
            # tensor = m_tot_k_1.cpu().squeeze()  # Remove the batch and channel dimensions
            # plt.figure(figsize=(20,20))
            # plt.imshow(tensor, cmap='cool')  # Plot the 64x64 image in grayscale
            # plt.savefig(f'm_tot_k_1{k}.png')
            
            m_BG_tot_k_1 = 1 - m_tot_k_1
            
            # tensor = m_BG_tot_k_1.cpu().squeeze()  # Remove the batch and channel dimensions
            # plt.figure(figsize=(20,20))
            # plt.imshow(tensor, cmap='cool')  # Plot the 64x64 image in grayscale
            # plt.savefig(f'm_BG_tot_k_1{k}.png')
            
            x_k_t0[k]=x_k_1_warp * m_tot_k_1 + x_k_t0[k-1] * m_BG_tot_k_1
            
            # tensor = (x_k_1_warp * m_tot_k_1).cpu().squeeze()  # Remove the batch dimension
            # fig, axs = plt.subplots(1, 4, figsize=(80, 20))
            # for i in range(tensor.shape[0]):
            #     axs[i].imshow(tensor[i], cmap='cool')  # Plot each 64x64 imag

            # plt.savefig(f'x_k_1_warp * m_tot_k_1{k}.png')
            
            # tensor = (x_k_t0[k-1] * m_BG_tot_k_1).cpu().squeeze()  # Remove the batch dimension
            # fig, axs = plt.subplots(1, 4, figsize=(80, 20))
            # for i in range(tensor.shape[0]):
            #     axs[i].imshow(tensor[i], cmap='cool')  # Plot each 64x64 imag

            # plt.savefig(f'x_k_t0[k-1] * m_BG_tot_k_1{k}.png')
            
            # tensor = x_k_t0[k].cpu().squeeze()  # Remove the batch dimension
            # fig, axs = plt.subplots(1, 4, figsize=(80, 20))
            # for i in range(tensor.shape[0]):
            #     axs[i].imshow(tensor[i], cmap='cool')  # Plot each 64x64 imag

            # plt.savefig(f'x_k_t0[k]{k}.png')
            
        del m_k_t0
        x_k_t0 = torch.stack(x_k_t0).squeeze() 

        # Perform forward process up to time T_1
        x_k_t1 = self.forward_loop(
            x_t0=x_k_t0,
            t0=timesteps[-t0 - 1].item(),
            t1=timesteps[-t1 - 1].item(),
            generator=generator,
        )

        # Perform backward process from time T_1 to 0
        b, l, d = prompt_embeds.size()
        prompt_embeds = prompt_embeds[:, None].repeat(1, video_length, 1, 1).reshape(b * video_length, l, d) #prompt_embeds[:, None] -- : la 1 dimension, nen:, None se new dimension 2

        self.timestep_counter = []
        self.scheduler = scheduler_copy
        x_k_0 = self.backward_loop(
            timesteps=timesteps[-t1 - 1 :],
            prompt_embeds=prompt_embeds,
            latents=x_k_t1,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            extra_step_kwargs=extra_step_kwargs,
            num_warmup_steps=0,
        )
        latents = x_k_0

        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
        torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        else:
            image = self.decode_latents(latents)
            # Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return TextToVideoPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)