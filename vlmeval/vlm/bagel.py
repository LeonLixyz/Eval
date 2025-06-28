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
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from vlmeval.vlm.base import BaseModel

# ---- import your own libraries ----
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from bagel.data.transforms import ImageTransform
from bagel.data.data_utils import add_special_tokens
from bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from bagel.modeling.qwen2 import Qwen2Tokenizer
from bagel.modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


class MyBagel(BaseModel):
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
        **infer_cfg,
    ):
        """
        Parameters
        ----------
        ckpt_dir   Path to the fine-tuned Bagel checkpoint folder
        base_model Path that holds llm_config.json, vit_config.json, ae.safetensors, tokenizer
        max_mem    Per-GPU memory budget given to `accelerate.infer_auto_device_map`
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
            cfg_renorm_type="text_channel",
        )
        self.hyper.update(infer_cfg)

    # --------------------------------------------------------------------- #
    #                              Inference                                #
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def generate_inner(self, msgs, dataset=None) -> str:
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
        prompt, images = self._msgs_to_prompt_imgs(msgs)
        current_input: List = [prompt] + images[:1]      # Bagel expects prompt then first image
        iteration = 0

        while True:
            # understanding pass
            out = self.inferencer.interleave_inference(
                current_input,
                understanding_output=True,
                system_prompt=None,
                **self.hyper,
            )[0]
            txt_step = self._extract_text(out)

            # check stop conditions
            if ("<|vision_start|>" not in out) or ("Final Answer" in out):
                return self._extract_final_answer(txt_step)

            # generation pass (image)
            gen_img = self.inferencer.interleave_inference(
                current_input + [txt_step],
                system_prompt=None,
                **self.hyper,
            )[0]                    # PIL.Image
            current_input = current_input + [txt_step, gen_img]
            iteration += 1

    # --------------------------------------------------------------------- #
    #                                Utils                                  #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _extract_text(raw: str) -> str:
        """Grab text between <|im_start|> and <|im_end|>."""
        return raw.split("<|im_end|>")[0].split("<|im_start|>")[1]

    @staticmethod
    def _extract_final_answer(step_text: str) -> str:
        """
        Works for outputs like:
        "... some reasoning ... Final Answer: B"
        """
        if "Final Answer" in step_text:
            return step_text.split("Final Answer")[-1].replace(":", "").strip()
        return step_text.strip()

    @staticmethod
    def _msgs_to_prompt_imgs(msgs) -> Tuple[str, List[Image.Image]]:
        prompt_parts, images = [], []
        for m in msgs:
            if m["type"] == "text":
                prompt_parts.append(m["value"])
            else:
                images.append(Image.open(Path(m["value"])).convert("RGB"))
        return "\n".join(prompt_parts), images

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
