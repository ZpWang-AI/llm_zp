from utils_zp import *
from data_utils import *
from llm_zp import *
from ModelSFT import *


pb = tqdm.tqdm()
LLaMAFactorySFTLora(
    model_name_or_path='/home/zhipang/pretrained_models/Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5',
    adapter_name_or_path=None,
    template='qwen2_vl',
    dataset='task_1.0',
    output_dir=DATA_DIR_SFT/'saves'/f'test_qwen_task1_sftlora',
).start(
    cuda_visible='2',
    llama_factory_dir=DATA_DIR_SFT.parent,
)
pb.update(1)

LLaMAFactoryMergeLora(
    model_name_or_path='/home/zhipang/pretrained_models/Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5',
    adapter_name_or_path='/home/zhipang/PhysicalDynamics/data/llama_factory_data/data/saves/qwen_raw_task_2.0_bin_20k_training',
    template='qwen2_vl',
    export_dir=DATA_DIR_SFT/'saves'/f'test_qwen_mergelora'
).start()
pb.update(1)