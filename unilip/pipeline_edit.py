from typing import List, Optional
from PIL import Image
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

class CustomEditPipeline:

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        multimodal_encoder: AutoModelForCausalLM,
        image_processor: AutoProcessor,
    ):
        super().__init__()
        self.multimodal_encoder = multimodal_encoder
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    @torch.no_grad()
    def __call__(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        crop_info: List[int] = [0, 0],
        original_size: List[int] = [1024, 1024],
        generator = None,
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        np_img = self._prepare_and_encode_inputs(
            inputs,
            do_classifier_free_guidance,
            generator=generator,
            guidance_scale=guidance_scale
        )
        images = self.numpy_to_pil(np_img)

        return images[0]

    def _prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        do_classifier_free_guidance: bool = False,
        generator=None,
        guidance_scale: float = 4.5,
    ):
        pos_text_prompt, neg_text_prompt, input_image = inputs
        pos_text_prompt = pos_text_prompt.replace(
                    "<image>",
                    f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}'
                )
        neg_text_prompt = neg_text_prompt.replace(
                    "<image>",
                    f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}'
                )
        resized_images = input_image.resize((448, 448))
        image_inputs = self.image_processor(resized_images, return_tensors="pt")
        image_prompt = image_inputs.pixel_values

        prompt = self.multimodal_encoder.generate_image(
            text=[pos_text_prompt, neg_text_prompt],
            pixel_values=image_prompt.cuda(),
            tokenizer=self.tokenizer,
            guidance_scale=guidance_scale
        )
        return prompt

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # Special case for grayscale (single channel) images.
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images
