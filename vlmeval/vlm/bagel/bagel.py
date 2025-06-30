"""
Bagel wrapper for VLMEvalKit.
Author: Leon (June 2025)

Usage
-----
python run.py \
  --model my_bagel \
  --data MathVista \
  --model-args "ckpt_dir=/path/ckpt base_model=/path/base max_mem=80GiB"
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple
from copy import deepcopy
import time
import base64
import io

import torch
from PIL import Image

from vlmeval.vlm.base import BaseModel

# ---- import your own libraries ----
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from .data.transforms import ImageTransform
from .data.data_utils import add_special_tokens, pil_img2rgb
from .modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from .modeling.qwen2 import Qwen2Tokenizer
from .modeling.autoencoder import load_ae
from .inferencer import InterleaveInferencer


class BagelVLM(BaseModel):
    """
    VLMEvalKit-compatible Vision-Language model that performs
    interleaved text–image reasoning and returns a **text answer**.
    """
    INTERLEAVE = True        # tell VLMEvalKit to send full message list

    # --------------------------------------------------------------------- #
    #                       Initialisation / Loading                        #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        ckpt_dir: str,
        base_model: str,
        max_mem: str = "80GiB",
        output_dir: str = ".",
        model_name: str = None,
        **infer_cfg,
    ):
        """
        Parameters
        ----------
        ckpt_dir   Path to the fine-tuned Bagel checkpoint folder
        base_model Path that holds llm_config.json, vit_config.json, ae.safetensors, tokenizer
        max_mem    Per-GPU memory budget given to `accelerate.infer_auto_device_map`
        output_dir Directory to save reasoning traces and outputs
        model_name Model name for organizing outputs
        infer_cfg  Any extra sampling hyper-parameters forwarded to InterleaveInferencer
        """
        super().__init__()
        (
            self.model,
            self.vae_model,
            self.tokenizer,
            self.vae_tf,
            self.vit_tf,
            self.new_ids,
        ) = self._load_bagel(ckpt_dir, base_model, max_mem)

        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_tf,
            vit_transform=self.vit_tf,
            new_token_ids=self.new_ids,
        )

        # default inference hyper-parameters (can be overridden via --model-args)
        self.hyper = dict(
            do_sample=True,
            text_temperature=0.7,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
        )
        self.hyper.update(infer_cfg)

        # Output directory setup
        self.base_output_dir = output_dir
        self.output_dir = output_dir
        
        # Extract model name from path if not provided
        if model_name is None:
            self.model_name = os.path.basename(ckpt_dir.rstrip('/'))
        else:
            self.model_name = model_name
            
        # Create temp directory for intermediate files
        self.temp_dir = tempfile.mkdtemp()
        
        os.makedirs(self.output_dir, exist_ok=True)

    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # --------------------------------------------------------------------- #
    #                              Inference                                #
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def generate(self, msgs, dataset=None) -> str:
        """
        Required by VLMEvalKit.

        Parameters
        ----------
        msgs  List[dict] where each dict has keys {"type": "text"|"image", "value": str}
              Images are local file paths.

        Returns
        -------
        answer  Plain string with the model's final answer.
        """
        # Parse VLMEvalKit message format
        prompt, images = self._parse_message(msgs)
        
        # Setup output directory for this query
        self.output_dir = os.path.join(self.base_output_dir, self.model_name)
        if dataset:
            self.output_dir = os.path.join(self.output_dir, dataset)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize reasoning trace storage
        reasoning_text = []
        reasoning_images = []
        
        # Initialize context once - efficient approach
        gen_context = self.inferencer.init_gen_context()
        cfg_text_context = None
        cfg_img_context = None

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Add initial prompt to context
            gen_context = self.inferencer.update_context_text(prompt, gen_context)
            
            if images:
                # Process the input image
                processed_image = self.inferencer.vae_transform.resize_transform(pil_img2rgb(images[0]))
                gen_context = self.inferencer.update_context_image(processed_image, gen_context, vae=False)
                
                # Set up CFG contexts
                cfg_text_context = deepcopy(gen_context)
                cfg_img_context = deepcopy(gen_context)

            iteration = 0
            full_response_text = ""
            
            while True:
                print(f"iteration: {iteration}")
                
                # Generate text response
                output_text = self.inferencer.gen_text(
                    gen_context,
                    do_sample=self.hyper['do_sample'],
                    temperature=self.hyper['text_temperature'],
                    max_length=4096
                )
                
                # Extract reasoning text
                txt_step = self._extract_text(output_text)
                reasoning_text.append(txt_step)
                full_response_text += txt_step + "\n"
                print(f"{txt_step}")
                
                # Check stop conditions
                should_stop = ('<answer>' in output_text) or ('Final Answer' in output_text) or ('<|vision_start|>' not in output_text)
                
                if should_stop:
                    # Update context with final response
                    gen_context = self.inferencer.update_context_text(txt_step, gen_context)
                    break
                
                # Update context with reasoning text
                gen_context = self.inferencer.update_context_text(txt_step, gen_context)
                cfg_img_context = self.inferencer.update_context_text(txt_step, cfg_img_context)
                
                # Generate image if vision_start token is present
                if "<|vision_start|>" in output_text:
                    # Update CFG text context
                    cfg_text_context = deepcopy(gen_context)
                    
                    # Generate image using current context
                    gen_img = self.inferencer.gen_image(
                        processed_image.size[::-1],  # image shape
                        gen_context,
                        cfg_text_precontext=cfg_text_context,
                        cfg_img_precontext=cfg_img_context,
                        cfg_text_scale=self.hyper['cfg_text_scale'],
                        cfg_img_scale=self.hyper['cfg_img_scale'],
                        cfg_interval=self.hyper['cfg_interval'],
                        timestep_shift=self.hyper['timestep_shift'],
                        num_timesteps=self.hyper['num_timesteps'],
                        cfg_renorm_min=self.hyper['cfg_renorm_min'],
                        cfg_renorm_type=self.hyper['cfg_renorm_type'],
                    )
                    
                    # Save and collect the generated image
                    reasoning_images.append(gen_img)
                    print(f"Generated reasoning image {len(reasoning_images)}")
                    
                    # Update context with generated image
                    processed_gen_image = self.inferencer.vae_transform.resize_transform(pil_img2rgb(gen_img))
                    gen_context = self.inferencer.update_context_image(processed_gen_image, gen_context, vae=False)
                    cfg_text_context = deepcopy(gen_context)
                
                iteration += 1
                print('-'*50)

        # Save comprehensive traces
        folder_name = self._sanitize_filename(prompt)
        self._save_to_folder(
            folder_name, 
            prompt, 
            images, 
            full_response_text, 
            reasoning_text,
            reasoning_images
        )
        
        # Extract and return final answer
        final_answer = self._extract_final_answer(full_response_text)
        return final_answer

    # --------------------------------------------------------------------- #
    #                                Utils                                  #
    # --------------------------------------------------------------------- #
    def _parse_message(self, message) -> Tuple[str, List[Image.Image]]:
        """
        Parse VLMEvalKit message format into prompt and images
        
        Args:
            message: VLMEvalKit message format
            
        Returns:
            (prompt_text, images_list)
        """
        images = []
        texts = []
        
        for item in message:
            if item['type'] == 'image':
                if isinstance(item['value'], str):
                    # Check if it's base64 data or file path
                    if item['value'].startswith('/9j/') or item['value'].startswith('data:image/') or len(item['value']) > 1000:
                        # Base64 data, decode it
                        if item['value'].startswith('data:image/'):
                            base64_data = item['value'].split(',')[1]
                        else:
                            base64_data = item['value']
                        
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_data))
                        images.append(image)
                    else:
                        # File path
                        images.append(Image.open(item['value']).convert('RGB'))
                else:
                    images.append(item['value'])
            elif item['type'] == 'text':
                texts.append(item['value'])
        
        prompt_text = '\n'.join(texts)
        
        # Handle <image> tokens
        image_count_in_text = prompt_text.count('<image>')
        
        # If no <image> tokens but images provided, add them
        if len(images) > 0 and image_count_in_text == 0:
            if len(images) == 1:
                prompt_text = f"{prompt_text}<image>"
            else:
                image_tokens = "<image>" * len(images)
                prompt_text = f"{prompt_text}{image_tokens}"
        
        return prompt_text, images

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Create a safe filename from text"""
        # Use timestamp-based naming for uniqueness
        timestamp = str(int(time.time() * 1000))[-8:]
        return f"query_{timestamp}"

    def _save_to_folder(self, folder_name: str, prompt_text: str, prompt_images: List[Image.Image], 
                       response_text: str, reasoning_text: List[str], reasoning_images: List[Image.Image]):
        """Save comprehensive traces to folder"""
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
        
        # Save reasoning steps
        with open(os.path.join(folder_path, "reasoning_steps.txt"), "w", encoding='utf-8') as f:
            for i, step in enumerate(reasoning_text):
                f.write(f"Step {i+1}:\n{step}\n\n{'='*50}\n\n")
        
        # Save reasoning images
        for i, img in enumerate(reasoning_images):
            img.save(os.path.join(folder_path, f"reasoning_image_{i+1}.png"))
        
        # Save metadata
        metadata = {
            "prompt": prompt_text,
            "num_prompt_images": len(prompt_images),
            "num_reasoning_steps": len(reasoning_text),
            "num_reasoning_images": len(reasoning_images),
            "final_answer": final_answer,
            "hyper_parameters": self.hyper
        }
        
        import json
        with open(os.path.join(folder_path, "metadata.json"), "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _extract_text(raw: str) -> str:
        """Grab text between <|im_start|> and <|im_end|>."""
        if '<|im_end|>' in raw and '<|im_start|>' in raw:
            return raw.split("<|im_end|>")[0].split("<|im_start|>")[1]
        return raw.strip()

    @staticmethod
    def _extract_final_answer(step_text: str) -> str:
        """
        Extract final answer from response text.
        Works for outputs like: "... some reasoning ... Final Answer: B"
        """
        import re
        
        # Look for various forms of "Final Answer:"
        patterns = [
            r"<answer>(.*?)</answer>",
            r"Final Answer:\s*(.*?)(?:\n\n|\Z)",
            r"Final answer:\s*(.*?)(?:\n\n|\Z)", 
            r"FINAL ANSWER:\s*(.*?)(?:\n\n|\Z)",
            r"Answer:\s*(.*?)(?:\n\n|\Z)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, step_text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no "Final Answer:" found, return the full response
        return step_text.strip()

    # --------------------------------------------------------------------- #
    #                       Weight / Tokeniser loading                      #
    # --------------------------------------------------------------------- #
    def _load_bagel(self, ckpt_dir: str, base_model: str, max_mem: str):
        """
        Re-implemented from Leon's standalone inference script,
        but wrapped into a function that returns everything ready for inference.
        """
        ckpt_file = "model_bf16.safetensors"
        ckpt_path = Path(ckpt_dir) / ckpt_file
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

        # --- 1. Configs ----------------------------------------------------
        llm_cfg = Qwen2Config.from_json_file(Path(base_model) / "llm_config.json")
        llm_cfg.qk_norm = True
        llm_cfg.tie_word_embeddings = False
        llm_cfg.layer_module = "Qwen2MoTDecoderLayer"

        vit_cfg = SiglipVisionConfig.from_json_file(Path(base_model) / "vit_config.json")
        vit_cfg.rope = False
        vit_cfg.num_hidden_layers -= 1

        vae_model, vae_cfg = load_ae(local_path=Path(base_model) / "ae.safetensors")

        bagel_cfg = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_cfg,
            vit_config=vit_cfg,
            vae_config=vae_cfg,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        # --- 2. Build skeleton (empty) ------------------------------------
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_cfg)
            vit_model = SiglipVisionModel(vit_cfg)
            model = Bagel(language_model, vit_model, bagel_cfg)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_cfg, meta=True)

        # --- 3. Tokeniser + special tokens --------------------------------
        tokenizer = Qwen2Tokenizer.from_pretrained(base_model)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # --- 4. Device map -------------------------------------------------
        n_gpu = torch.cuda.device_count()
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem for i in range(n_gpu)},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            dtype=torch.bfloat16,
        )

        # Some modules must live on the same GPU
        same_device = [
            'language_model.model.embed_tokens',
            'time_embedder', 'latent_pos_embed',
            'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed',
        ]
        target_device = device_map.get(same_device[0], "cuda:0")
        for m in same_device:
            device_map[m] = target_device

        # --- 5. Load weights (bf16 on device) ------------------------------
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(ckpt_path),
            device_map=device_map,
            offload_buffers=False,
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()

        # --- 6. Image transforms ------------------------------------------
        vae_tf = ImageTransform(1024, 512, 16)
        vit_tf = ImageTransform(980, 512, 14)

        return model, vae_model, tokenizer, vae_tf, vit_tf, new_token_ids
