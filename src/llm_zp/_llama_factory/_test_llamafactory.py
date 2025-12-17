from utils_zp import *
from llm_zp import *

'''
hf download --resume-download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen/Qwen3-VL-2B-Instruct
huggingface-cli download --resume-download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen/Qwen3-VL-2B-Instruct
'''

LLAMAFACTORY_DIR = path('/root/LLaMA-Factory')
model = 'Qwen/Qwen3-VL-2B-Instruct'
template = 'qwen3_vl'
output_dir = LLAMAFACTORY_DIR/'saves'/'test_qwen_sftlora'

pb = tqdm.tqdm()
if 1:
    LLaMAFactorySFTLora(
        model_name_or_path=model,
        adapter_name_or_path='/root/LLaMA-Factory/saves/test_qwen_sftlora',
        template=template,
        dataset='alpaca_en_demo',
        output_dir=output_dir,

        flash_attn='fa2'
    ).start(
        cuda_visible='0,1',
        llama_factory_dir=LLAMAFACTORY_DIR,
    )
    pb.update(1)

if 1:
    LLaMAFactoryMergeLora(
        model_name_or_path=model,
        adapter_name_or_path=output_dir,
        template='qwen2_vl',
        export_dir=LLAMAFACTORY_DIR/'saves'/f'test_qwen_mergelora'
    ).start()
    pb.update(1)