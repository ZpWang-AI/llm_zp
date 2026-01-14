from utils_zp import *
from ._llm_base_class import LLMBaseClass_zp

"""
https://huggingface.co/collections/Qwen/qwen3
"""


class Qwen3(LLMBaseClass_zp):
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
            
            from transformers import AutoModelForCausalLM
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_or_model_path, 
                **model_kwargs
            ).eval()
            self._self_model_merge_adapter()
        
        return self._model

    def __call__(self, conversation:list=None, query:str=None):
        if conversation is None:
            assert query is not None
            conversation = [{"role": "user", "content": query}]
        else:
            assert query is None
        return super().__call__(conversation=conversation)


