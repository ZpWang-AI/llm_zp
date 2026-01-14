from utils_zp import *
from ._llm_base_class import LLMBaseClass_zp

"""
https://huggingface.co/OpenGVLab/InternVL3_5-8B

"""


class InternVL3_5(LLMBaseClass_zp):
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
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_or_model_path, 
                **model_kwargs
            ).eval()
            self._self_model_merge_adapter()

        return self._model
    
