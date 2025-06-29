import torch
import json
import os
import re
import string
import pandas as pd
from typing import List, Dict, Any, Union
from vlmeval.vlm.base import BaseModel
from vlmeval.smp import *
from transformers import ChameleonConfig, ChameleonProcessor, ChameleonForConditionalGenerationWithCFG
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.image_transforms import to_pil_image
from PIL import Image
import tempfile
import shutil


class StopAtSpecificTokenCriteria(StoppingCriteria):
    """Stop generation when a specific token is generated"""
    
    def __init__(self, stop_token_id, device):
        self.stop_token_id = stop_token_id
        self.device = device
    
    def __call__(self, input_ids, scores, **kwargs):
        return (input_ids[0, -1] == self.stop_token_id).item()


class AnoleVLM(BaseModel):
    """VLMEvalKit wrapper for Chameleon interleaved generation model"""
    
    INTERLEAVE = True  # This model supports interleaved text-image inputs
    
    def __init__(
        self, 
        model_path: str, 
        mode: str = "general",
        cfg_type: str = "normal",
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_length: int = 12288,
        max_images: int = 4,
        output_dir: str = ".",  # Use current directory
        model_name: str = None,
        **kwargs
    ):
        """
        Initialize Chameleon model for VLMEvalKit
        
        Args:
            model_path: Path to the Chameleon model
            mode: Generation mode ("general", "image_critique", "object_thoughts")
            cfg_type: CFG type ("normal", "obj", "full")
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_length: Maximum generation length
            max_images: Maximum images to generate
        """
        self.model_path = model_path
        self.mode = mode
        self.cfg_type = cfg_type
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.max_images = max_images
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model components
        self.config = ChameleonConfig.from_pretrained(model_path)
        self.config.attn_implementation = "flash_attention_2"
        
        self.processor = ChameleonProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"
        
        self.model = ChameleonForConditionalGenerationWithCFG.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(self.device)
        
        # Get special tokens
        self.boi_token_id = self.config.boi_token_id
        self.eoi_token_id = self.config.eoi_token_id
        self.eos_token_id = self.config.eos_token_id
        self.pad_token_id = 1
        
        # Image vocabulary for filtering
        self.image_conditioned_allowed = set([i for i in range(4, 8196)]) | {
            self.config.bos_token_id,
            self.boi_token_id,
            self.eoi_token_id,
        }
        
        # Setup initial CFG
        self.model.setup_cfg(
            guidance_scale_full=2.0,
            guidance_scale_image=1.2,
            guidance_scale_negative=0.0,
            guidance_scale_original_prompt=5.0,
            config=self.config,
            cfg_config="no"
        )
        
        # Create temp directory for image outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # VLMEvalKit compatibility
        self.version = kwargs.get('version', 'V1.0')
        
        # Extract model name from path if not provided
        if model_name is None:
            self.model_name = os.path.basename(model_path.rstrip('/'))
        else:
            self.model_name = model_name

        # Store the base output directory
        self.base_output_dir = output_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Create a safe filename from text"""
        # Just use a simple sequential naming scheme
        import time
        timestamp = str(int(time.time() * 1000))[-8:]  # Last 8 digits of timestamp
        return f"query_{timestamp}"
    
    def _decode_image_from_tokens(self, image_tokens: List[int]) -> Image.Image:
        """Decode image tokens to PIL Image"""
        try:
            token_tensor = torch.tensor([image_tokens], device=self.device, dtype=torch.long)
            pixel_values = self.model.model.decode_image_tokens(token_tensor)
            images = self.processor.postprocess_pixel_values(pixel_values)
            image = to_pil_image(images[0].detach().cpu())
            return image
        except Exception as e:
            print(f"Error decoding image: {e}")
            return Image.new('RGB', (512, 512), color='gray')
    
    def _save_to_folder(self, folder_name: str, prompt_text: str, prompt_images: List[Image.Image], 
                       response_text: str, response_tokens: List[int]):
        """Save prompt and response to folder"""
        # Create folder
        folder_path = os.path.join(self.output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Save prompt text
        with open(os.path.join(folder_path, "prompt.txt"), "w", encoding='utf-8') as f:
            f.write(prompt_text)
        
        # Save prompt images
        for i, img in enumerate(prompt_images):
            img.save(os.path.join(folder_path, f"prompt_image_{i}.png"))
        
        # Save full response text
        with open(os.path.join(folder_path, "response_full.txt"), "w", encoding='utf-8') as f:
            f.write(response_text)
        
        # Save final answer only
        final_answer = self._extract_final_answer(response_text)
        with open(os.path.join(folder_path, "response.txt"), "w", encoding='utf-8') as f:
            f.write(final_answer)
        
        # Extract and save response images
        self._extract_and_save_response_images(response_tokens, folder_path)
    
    def _extract_and_save_response_images(self, tokens: List[int], folder_path: str):
        """Extract images from response tokens and save them"""
        current_tokens = []
        image_count = 0
        in_image = False
        
        for token in tokens:
            if token == self.boi_token_id:
                in_image = True
                current_tokens = []
            elif token == self.eoi_token_id and in_image:
                if len(current_tokens) == 1024:  # Valid image
                    image = self._decode_image_from_tokens(current_tokens)
                    image.save(os.path.join(folder_path, f"response_image_{image_count}.png"))
                    image_count += 1
                in_image = False
                current_tokens = []
            elif in_image:
                current_tokens.append(token)
    
    def _extract_final_answer(self, response_text: str) -> str:
        """Extract text after 'Final Answer:' or return full response if not found"""
        # Look for various forms of "Final Answer:"
        patterns = [
            r"Final Answer:\s*(.*?)(?:\n\n|\Z)",
            r"Final answer:\s*(.*?)(?:\n\n|\Z)", 
            r"FINAL ANSWER:\s*(.*?)(?:\n\n|\Z)",
            r"Answer:\s*(.*?)(?:\n\n|\Z)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no "Final Answer:" found, return the full response
        return response_text.strip()
     
    def _parse_message(self, message: Union[List[str], List[Dict[str, str]]]) -> tuple:
        """
        Parse VLMEvalKit message format into images and text
        
        Returns:
            (images, text_prompt)
        """
        images = []
        texts = []
        
        if isinstance(message[0], dict):
            # print(f"DEBUG: Parsing message: {message}")
            # Dict format: [{'type': 'image', 'value': 'path'}, {'type': 'text', 'value': 'text'}]
            for item in message:
                if item['type'] == 'image':
                    if isinstance(item['value'], str):
                        # Check if it's base64 data or a file path
                        if item['value'].startswith('/9j/') or item['value'].startswith('data:image/') or len(item['value']) > 1000:
                            # It's base64 data, decode it
                            import base64
                            import io
                            if item['value'].startswith('data:image/'):
                                # Remove data:image/jpeg;base64, prefix
                                base64_data = item['value'].split(',')[1]
                            else:
                                base64_data = item['value']
                            
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_data))
                            images.append(image)
                            # save the image to current directory with timestamp
                            # image.save(f"image_{time.time()}.png")
                        else:
                            # It's a file path
                            images.append(Image.open(item['value']))
                    else:
                        images.append(item['value'])
                elif item['type'] == 'text':
                    texts.append(item['value'])
        else:
            # List format: ['image.jpg', 'What is this?']
            for item in message:
                if isinstance(item, str):
                    if item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        images.append(Image.open(item))
                    else:
                        texts.append(item)
                elif hasattr(item, 'size'):  # PIL Image
                    images.append(item)
        
        text_prompt = ' '.join(texts)
        
        # Count <image> tokens in the text
        image_count_in_text = text_prompt.count('<image>')
        
        # If there are <image> tokens in text but no images provided, or mismatch
        if image_count_in_text > 0:
            if len(images) == 0:
                print(f"Warning: Found {image_count_in_text} <image> tokens in text but no images provided")
            elif len(images) != image_count_in_text:
                print(f"Warning: Found {image_count_in_text} <image> tokens in text but {len(images)} images provided")
        
        # If no <image> tokens in text but images are provided, we need to add the image token
        elif len(images) > 0 and image_count_in_text == 0:
            # print(f"Warning: Found {len(images)} images but no <image> tokens in text")
            # For single image, add <image> at the end of the prompt
            if len(images) == 1:
                text_prompt = f"{text_prompt}<image>"
            else:
                # For multiple images, add <image> tokens at the end
                image_tokens = "<image>" * len(images)
                text_prompt = f"{text_prompt}{image_tokens}"
        
        return images, text_prompt

    def _prepare_cfg_batch(self, token_ids, cfg_type="normal"):
        """Prepare batch for CFG by creating multiple conditions"""
        negative_prompt = "text in the image, text, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        negative_tokens = self.processor.tokenizer.encode(negative_prompt, add_special_tokens=False)
        
        batch_token_ids = []
        
        if cfg_type == "normal":
            batch_token_ids.append(token_ids)
            batch_token_ids.append([self.boi_token_id])
            image_only_tokens = [tok for tok in token_ids if tok in self.image_conditioned_allowed]
            if not image_only_tokens or image_only_tokens[-1] != self.boi_token_id:
                image_only_tokens.append(self.boi_token_id)
            batch_token_ids.append(image_only_tokens)
            
        elif cfg_type == "obj":
            batch_token_ids.append(token_ids)
            batch_token_ids.append([self.boi_token_id])
            batch_token_ids.append(negative_tokens + [self.boi_token_id])
                
        elif cfg_type == "full":
            batch_token_ids.append(token_ids)
            image_only_tokens = [tok for tok in token_ids if tok in self.image_conditioned_allowed]
            if not image_only_tokens or image_only_tokens[-1] != self.boi_token_id:
                image_only_tokens.append(self.boi_token_id)
            batch_token_ids.append(image_only_tokens)
            batch_token_ids.append([self.boi_token_id])
            batch_token_ids.append(negative_tokens + [self.boi_token_id])
            batch_token_ids.append([self.boi_token_id])
        
        # Pad sequences
        max_len = max(len(seq) for seq in batch_token_ids)
        attention_masks = []
        
        for i, seq in enumerate(batch_token_ids):
            padding_length = max_len - len(seq)
            if padding_length > 0:
                batch_token_ids[i] = [self.pad_token_id] * padding_length + seq
                attention_masks.append([0] * padding_length + [1] * len(seq))
            else:
                attention_masks.append([1] * len(seq))
        
        input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)
        
        return input_ids, attention_mask
    
    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens, handling both text and image tokens"""
        text_parts = []
        current_text_tokens = []
        in_image = False
        image_count = 0
        
        for token in tokens:
            if token == self.boi_token_id:
                # Decode accumulated text tokens
                if current_text_tokens:
                    text = self.processor.tokenizer.decode(current_text_tokens, skip_special_tokens=True)
                    text_parts.append(text)
                    current_text_tokens = []
                in_image = True
            elif token == self.eoi_token_id and in_image:
                # End of image
                text_parts.append(f"[Generated Image {image_count}]")
                image_count += 1
                in_image = False
            elif not in_image and token not in [self.eos_token_id, self.pad_token_id]:
                current_text_tokens.append(token)
        
        # Decode any remaining text tokens
        if current_text_tokens:
            text = self.processor.tokenizer.decode(current_text_tokens, skip_special_tokens=True)
            text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def generate(self, message: Union[List[str], List[Dict[str, str]]], dataset: str = None) -> str:
        """
        Generate response for VLMEvalKit
        
        Args:
            message: Input in VLMEvalKit format
            dataset: Dataset name (optional, for custom prompts)
            
        Returns:
            Generated text response
        """
        # Parse input
        images, text_prompt = self._parse_message(message)
        
        # Reset output directory to base and create dataset subfolder if needed
        self.output_dir = os.path.join(self.base_output_dir, self.model_name)
        if dataset:
            self.output_dir = os.path.join(self.output_dir, dataset)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Prepare input based on mode
        if self.mode == "image_critique":
            full_prompt = f"Generate an image based on the given prompt. Then analyze whether the image matches the prompt, and generate a better image based on your analysis. The prompt is: {text_prompt}"
            prompt_tokens = self.processor.tokenizer.encode(full_prompt, add_special_tokens=False) + [8710, 8197]
            original_prompt_tokens = self.processor.tokenizer.encode(text_prompt, add_special_tokens=False)
        elif self.mode == "object_thoughts":
            full_prompt = f"Generate the objects in the prompt step by step, and then generate the complete image. The prompt is: {text_prompt}"
            prompt_tokens = self.processor.tokenizer.encode(full_prompt, add_special_tokens=False) + [8710]
            original_prompt_tokens = self.processor.tokenizer.encode(text_prompt, add_special_tokens=False)
        else:  # general mode
            # Handle input images if provided
            if images:
                text_prompt = "You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of thought. Think step by step and use visual aids to enhance your problem-solving. Provide your final conclusion clearly in the format of 'Final Answer: <answer here>'/n/n" + text_prompt
                inputs = self.processor(text_prompt, images=images, padding=False, 
                                      return_tensors="pt", return_for_text_completion=True)
                inputs = inputs.to(self.device, dtype=torch.bfloat16)
                input_ids = inputs['input_ids']
                
                # Process image tokens
                pixel_values = inputs["pixel_values"]
                image_tokens = self.model.model.get_image_tokens(pixel_values)
                special_image_mask = input_ids == 8711  # Image token ID
                image_tokens = image_tokens.to(input_ids.device, input_ids.dtype)
                input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)
                
                prompt_tokens = input_ids[0].tolist() + [8710]
            else:
                prompt_tokens = self.processor.tokenizer.encode(text_prompt, add_special_tokens=False) + [8710]
            
            original_prompt_tokens = prompt_tokens.copy()
        
        # Generate interleaved output
        all_tokens = prompt_tokens.copy()
        num_images_generated = 0
        
        while len(all_tokens) < self.max_length and num_images_generated < self.max_images:
            current_input_ids = torch.tensor([all_tokens], device=self.device)
            
            # Setup stopping criteria
            stop_at_boi = StopAtSpecificTokenCriteria(self.boi_token_id, self.device)
            stop_at_eos = StopAtSpecificTokenCriteria(self.eos_token_id, self.device)
            
            # Generate text
            self.model.cfg_config = "no"
            
            if self.mode == "image_critique" and num_images_generated == 0:
                new_tokens = []
            else:
                text_output = self.model.generate(
                    input_ids=current_input_ids,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    stopping_criteria=StoppingCriteriaList([stop_at_boi, stop_at_eos]),
                    multimodal_generation_mode="interleaved-text-image",
                    pad_token_id=self.pad_token_id
                )
                new_tokens = text_output[0][len(all_tokens):].tolist()
            
            all_tokens.extend(new_tokens)
            
            # Check stopping conditions
            if all_tokens[-1] == self.eos_token_id:
                break
            
            # Generate image if needed
            if all_tokens[-1] == self.boi_token_id:
                # Determine CFG type
                actual_cfg_type = self.cfg_type
                if self.mode == "object_thoughts":
                    actual_cfg_type = "obj" if num_images_generated < 2 else "full"
                elif self.mode == "image_critique":
                    actual_cfg_type = "full"
                
                # Enable CFG for image
                self.model.cfg_config = actual_cfg_type
                
                # Prepare CFG batch
                cfg_input_ids, cfg_attention_mask = self._prepare_cfg_batch(
                    all_tokens, cfg_type=actual_cfg_type
                )
                
                # Generate image tokens
                image_output = self.model.generate(
                    input_ids=cfg_input_ids,
                    attention_mask=cfg_attention_mask,
                    max_new_tokens=1026,
                    temperature=self.temperature,
                    do_sample=True,
                    multimodal_generation_mode="image-only",
                    pad_token_id=self.pad_token_id
                )
                
                new_image_tokens = image_output[0][len(cfg_input_ids[0]):].tolist()[:1025]
                all_tokens.extend(new_image_tokens)
                num_images_generated += 1
                
                # Disable CFG for next text
                self.model.cfg_config = "no"
        
        # Decode the complete generation
        response_tokens = all_tokens[len(prompt_tokens):]
        response_text = self._decode_tokens(response_tokens)
        
        # Save to folder
        folder_name = self._sanitize_filename(text_prompt)
        self._save_to_folder(folder_name, text_prompt, images, response_text, response_tokens)
        
        # Extract final answer if present
        # response_text = self._extract_final_answer(response_text)
        
        return response_text
    
    def use_custom_prompt(self, dataset: str) -> bool:
        """Check if custom prompt should be used for dataset"""
        # You can add specific datasets that need custom prompts
        custom_prompt_datasets = ['MMBench', 'MathVista', 'ScienceQA', 'MMVet', 'MathVerse']
        return dataset in custom_prompt_datasets
    
    def build_prompt(self, line: Dict[str, Any], dataset: str = None) -> List[Union[str, Dict[str, str]]]:
        """Build custom prompt for specific datasets"""
        # Extract components from the dataset line
        image = line.get('image', '')
        question = line.get('question', '')
        options = line.get('options', [])
        
        return [{'type': 'image', 'value': image}, {'type': 'text', 'value': question}]


# Register the model in vlmeval/config.py
# Add this to the supported_VLM dictionary:
# 'chameleon_interleaved': partial(ChameleonVLM, model_path='path/to/model'),
# 'chameleon_critique': partial(ChameleonVLM, model_path='path/to/model', mode='image_critique'),
# 'chameleon_objects': partial(ChameleonVLM, model_path='path/to/model', mode='object_thoughts'),