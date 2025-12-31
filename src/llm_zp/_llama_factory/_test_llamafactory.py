from utils_zp import *
from llm_zp import *

'''
hf download --resume-download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen/Qwen3-VL-2B-Instruct
huggingface-cli download --resume-download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen/Qwen3-VL-2B-Instruct
'''

LLAMAFACTORY_DIR = path('/root/autodl-fs/LLaMA-Factory')
dataset_name = 'alpaca_en_demo'
model = 'Qwen/Qwen3-VL-2B-Instruct'
template = 'qwen3_vl'
adapter_name_or_path = None

LLAMAFACTORY_DIR = path('/root/autodl-fs/_fs_data/PhyDy_data/data/llama_factory_data')
dataset_name = 'task_2.0_bin_20k'
model = '/root/autodl-fs/_fs_data/pretrained_models/Qwen2.5-VL-7B-Instruct'
template = 'qwen2_vl'
adapter_name_or_path = '/root/autodl-fs/PhysicalDynamics/data/llama_factory_data/saves/annotation_model_ckpt_lora'

output_dir = LLAMAFACTORY_DIR/'saves'/'test_qwen_sftlora'

pb = tqdm.tqdm(desc='llama factory test')
if 1:
    LLaMAFactorySFTLora(
        model_name_or_path=model,
        adapter_name_or_path=adapter_name_or_path,
        template=template,
        dataset=dataset_name,
        output_dir=output_dir,
        max_samples=20,

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
        template=template,
        export_dir=LLAMAFACTORY_DIR/'saves'/f'test_qwen_mergelora'
    ).start()
    pb.update(1)