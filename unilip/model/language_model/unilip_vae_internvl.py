from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F


import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from unilip.model.unilip_vae_internvl import UniLIP_VAE_InternVL_MetaModel, UniLIP_VAE_InternVL_MetaForCausalLM

from transformers import InternVLConfig, InternVLModel, InternVLForConditionalGeneration

from unilip.constants import UND_IMAGE_TOKEN_IDX

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import numpy_to_pil
import numpy as np
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.pipelines.sana.pipeline_sana import SanaPipeline

class UniLIP_VAE_InternVLConfig(InternVLConfig):
    model_type = "unilip_vae_internvl"


class UniLIP_VAE_InternVLModel(UniLIP_VAE_InternVL_MetaModel, InternVLModel):
    config_class = UniLIP_VAE_InternVLConfig

    def __init__(self, config: InternVLConfig):
        super(UniLIP_VAE_InternVLModel, self).__init__(config)


class UniLIP_VAE_InternVLForCausalLM(InternVLForConditionalGeneration, UniLIP_VAE_InternVL_MetaForCausalLM):
    config_class = UniLIP_VAE_InternVLConfig

    def __init__(self, config):
        InternVLForConditionalGeneration.__init__(self, config)
        config.model_type = "unilip_internvl"
        self.model = UniLIP_VAE_InternVLModel(config)   
        
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ids: Optional[list] = None,
        i_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gen_image: Optional[torch.FloatTensor] = None,
        und_image: Optional[torch.FloatTensor] = None,
        grid_thw: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                latents,
                bidr_attention_mask,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                gen_image,
                und_image,
                grid_thw,
                i_s_pos,
                image_sizes
            )

        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0
        outputs = self.model.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=return_dict,
            use_cache=False
        )
        
        hidden_states = outputs.hidden_states[-1]
        logits = None
        
        total_loss = None
        if labels is not None:
            img_loss_funct = torch.nn.MSELoss()
            img_hidden_states = self.model.llm_connector(
                attention_mask=bidr_attention_mask,
                position_ids=position_ids,
                inputs_embeds=hidden_states,
                output_hidden_states=True,
                return_dict=return_dict,
                use_cache=False
            ).hidden_states[-1]
            img_hidden_states = self.get_model().projector(img_hidden_states)
            if latents is None:
                img_loss = img_loss_funct(img_hidden_states, torch.clone(img_hidden_states.detach()))
            else:
                bsz = latents.shape[0]
                dtype = latents.dtype
                noise = torch.randn_like(latents, device=latents.device)
                u = compute_density_for_timestep_sampling(weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0)
                indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
                sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                noise_pred = self.get_model().dit(
                    noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=img_hidden_states,
                    encoder_attention_mask=attention_mask,
                    return_dict=False
                )[0]
                target = noise - latents

                img_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            print(f"img loss {img_loss}")
            total_loss = img_loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                img_indicator,
                _
            ) = self.prepare_inputs_labels_for_understanding(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        tokenizer: AutoTokenizer,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_var: Optional[float] = None,
        generator=None,
        guidance_scale: float = 4.5,
    ):  

        dit_path = self.model.config.dit_path
        scheduler = DPMSolverMultistepScheduler.from_pretrained(dit_path, subfolder="scheduler")

        N_QUERY = self.get_n_query()            
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        device = self.get_model().device
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)  # B x N

        text_embeds = self.get_model().language_model.embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)


        if pixel_values is not None:
            und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
            pixel_values = pixel_values.type(self.vision_tower.dtype)
            vision_feature_layer = self.config.vision_feature_layer
            vision_feature_select_strategy = self.config.vision_feature_select_strategy
            und_image_embeds = self.model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=None,
            )

            text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device).repeat(2,1,1).flatten(0,1)

        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0])], dim=1).int()
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        outputs = self.model.language_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask.bool(),
            position_ids= position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]
        attention_mask = attention_mask.bool()
        bidr_attention_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
        bidr_attention_mask = bidr_attention_mask.unsqueeze(1)
        bidr_attention_mask = (1-bidr_attention_mask.float())*-100000
        img_hidden_states = self.model.llm_connector(
            inputs_embeds=hidden_states,
            attention_mask=bidr_attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        img_hidden_states = self.get_model().projector(img_hidden_states) 
        steps = 20
        print("steps, guiance scale", steps, guidance_scale)
        # null first
        bsz = img_hidden_states.shape[0]
        img_hidden_states = torch.cat([img_hidden_states[bsz//2:], img_hidden_states[:bsz//2]])
        output_img = self.sample_images(img_hidden_states, scheduler, encoder_attention_mask=attention_mask, generator=generator, num_inference_steps=steps, guidance_scale=guidance_scale)
        output_img = self.model.dcae.decode(output_img.float()/ self.model.dcae.config.scaling_factor).sample
        
        output_img = ((output_img[0].permute(1,2,0).clamp(-1,1).float().cpu().numpy() + 1)/2)
        return output_img

    def sample_images(
        self,
        img_hidden_states,
        scheduler,
        encoder_attention_mask=None,
        guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        **kwargs,
    ):

        device = img_hidden_states.device
        dtype = img_hidden_states.dtype

        img_hidden_states_input = img_hidden_states
        # here already is cfg feature
        batch_size = img_hidden_states.shape[0]//2
        latent_size = 16
        latent_channels = 32

        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # set step values
        if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        else:
            scheduler.set_timesteps(num_inference_steps)

        # Repeat z_latents and conditions for each image per prompt
        img_hidden_states_input = img_hidden_states_input.repeat_interleave(num_images_per_prompt, dim=0)
        for t in scheduler.timesteps:
            latent_model_input = latents.repeat(2, 1, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(img_hidden_states_input.dtype)
            # predict noise model_output
            noise_pred = self.get_model().dit(
                latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                encoder_hidden_states=img_hidden_states_input,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

        return latents

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("unilip_vae_internvl", UniLIP_VAE_InternVLConfig)
AutoModelForCausalLM.register(UniLIP_VAE_InternVLConfig, UniLIP_VAE_InternVLForCausalLM)
