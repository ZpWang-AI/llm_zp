from utils_zp import *
from ._llm_base_class import _LLMBaseClass

"""
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

pip install accelerate
pip install qwen_vl_utils
"""


class Qwen2_5_VL(_LLMBaseClass):
    @property
    def model(self):
        if self._model is None:
            if self.model_load_mode == 'bf16':
                model_kwargs = {
                    'torch_dtype': torch.bfloat16,
                    'low_cpu_mem_usage': True,
                    'trust_remote_code': True,
                    'device_map': 'auto',
                    # 'device_map': 'cuda',
                    # 'use_flash_attn': True,
                    # 'attn_implementation': 'flash_attention_2'
                }
            else:
                raise Exception(f'wrong mode: {self.model_load_mode}')
            
            from transformers import AutoModelForImageTextToText
            if self._model is None:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_or_model_path, 
                    **model_kwargs
                ).eval()
        
        return self._model
    
    def tokenize(self, conversation, num_frames = None, fps = None):
        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(conversation, return_video_kwargs=True)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            fps=fps,
            **video_kwargs
        )
        return inputs
    