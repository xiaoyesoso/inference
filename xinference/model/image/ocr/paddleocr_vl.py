# Copyright 2022-2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import PIL.Image
import torch

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class PaddleOCRVLModel(OCRModel):
    """PaddleOCR-VL model for OCR, table recognition, formula recognition, chart recognition, seal recognition and text spotting.

    Supports both PaddleOCR-VL (initial release) and PaddleOCR-VL-1.6, which
    share the same model architecture and inference protocol but differ in
    weights and required transformers version.
    """

    required_libs = ("transformers",)

    SUPPORTED_MODEL_NAMES = {"PaddleOCR-VL", "PaddleOCR-VL-1.6"}

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name in cls.SUPPORTED_MODEL_NAMES

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        # model info when loading
        self._model = None
        self._processor = None
        # info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(f"Loading PaddleOCR-VL model from {self._model_path}")

        try:
            # Determine device and dtype
            if self._device == "cpu":
                device = "cpu"
                dtype = torch.float32
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self._model_path, trust_remote_code=True
            )

            # Load model
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self._model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
                .to(device)
                .eval()
            )

            logger.info(
                f"PaddleOCR-VL model loaded successfully on {device} with dtype {dtype}"
            )
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR-VL model: {e}")
            raise

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        **kwargs,
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        Perform OCR, table recognition, formula recognition, or chart recognition.

        Args:
            image: PIL Image or list of PIL Images
            **kwargs: Additional parameters including:
                - task: Task type ('ocr', 'table', 'formula', 'chart',
                  'spotting', 'seal'), default: 'ocr'. The 'spotting' and
                  'seal' tasks are only meaningful for PaddleOCR-VL-1.6+.
                - prompt: Custom prompt (optional, overrides task-based prompt)
                - max_new_tokens: Maximum number of tokens to generate (default: 1024)
                - return_dict: Whether to return a dictionary with metadata (default: False)

        Returns:
            OCR results as string, list of strings, or dict
        """
        logger.info("PaddleOCR-VL kwargs: %s", kwargs)

        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Extract parameters
        task = kwargs.get("task", "ocr")
        custom_prompt = kwargs.get("prompt", None)
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        return_dict = kwargs.get("return_dict", False)

        # Define task prompts. PaddleOCR-VL-1.6 introduces additional
        # 'spotting' and 'seal' tasks; older PaddleOCR-VL releases only
        # ship the four base tasks. Sending an unknown prompt to a model
        # that does not understand it will simply produce a less accurate
        # result, so we expose all prompts uniformly and let the underlying
        # model handle it.
        PROMPTS = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        }

        # Use custom prompt if provided, otherwise use task-based prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = PROMPTS.get(task, PROMPTS["ocr"])

        # Handle single image input
        if isinstance(image, PIL.Image.Image):
            result = self._process_single(image, prompt, max_new_tokens, task)
            if return_dict:
                return {
                    "text": result,
                    "model": "paddleocr-vl",
                    "task": task,
                    "success": True,
                }
            return result

        # Handle batch image input
        elif isinstance(image, list):
            results = [
                self._process_single(img, prompt, max_new_tokens, task) for img in image
            ]
            if return_dict:
                return {
                    "text": results,
                    "model": "paddleocr-vl",
                    "task": task,
                    "success": True,
                    "num_images": len(results),
                }
            return results

        else:
            raise ValueError("Input must be a PIL Image or list of PIL Images")

    def _process_single(
        self,
        image: PIL.Image.Image,
        prompt: str,
        max_new_tokens: int,
        task: str = "ocr",
    ) -> str:
        """Process a single image with the given prompt.

        For the ``spotting`` task on small images (<1500px on both
        dimensions), we follow the official PaddleOCR-VL-1.6 reference and
        upscale by 2x using LANCZOS to improve detection accuracy. The
        underlying processor's ``max_pixels`` is also bumped from the
        default ~1M pixels to 1605632 (2048 * 28 * 28) for the spotting
        task to keep the longer side from being aggressively downsampled.
        """
        # Ensure model and processor are loaded
        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        # Convert image to RGB if needed
        if image.mode in ["RGBA", "CMYK"]:
            image = image.convert("RGB")

        # Spotting upscale: small images benefit from 2x upscaling so that
        # the vision tower sees enough pixels for fine-grained character
        # localization.
        spotting_upscale_threshold = 1500
        if (
            task == "spotting"
            and image.width < spotting_upscale_threshold
            and image.height < spotting_upscale_threshold
        ):
            try:
                resample_filter = PIL.Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = PIL.Image.LANCZOS
            image = image.resize((image.width * 2, image.height * 2), resample_filter)

        # Decide max_pixels: spotting needs more visual detail, every other
        # task uses the processor's default (~1M pixels = 1280 * 28 * 28).
        max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28

        # Get device
        device = next(self._model.parameters()).device

        # Prepare messages in the format expected by PaddleOCR-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template. ``images_kwargs`` is forwarded into the
        # underlying image processor so we can override the max_pixels per
        # task (PaddleOCR-VL-1.6 reference uses this exact key).
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                images_kwargs={
                    "size": {
                        "shortest_edge": self._processor.image_processor.min_pixels,
                        "longest_edge": max_pixels,
                    }
                },
            ).to(device)
        except TypeError:
            # Older PaddleOCR-VL processors may not accept ``images_kwargs``;
            # fall back to the plain call.
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

        # Generate
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode output
        # Slice to remove input prompt from output
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        result = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return result
