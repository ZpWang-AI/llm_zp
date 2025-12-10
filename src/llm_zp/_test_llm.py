from utils_zp import *
from llm_zp import Qwen2_5_VL, InternVL3_5, LLaVA_NeXT_Video

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

model = Qwen2_5_VL('/home/zhipang/pretrained_models/Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
model = InternVL3_5('/home/zhipang/pretrained_models/InternVL3_5-8B-HF')
model = LLaVA_NeXT_Video('/home/zhipang/pretrained_models/LLaVA-NeXT-Video-7B-hf')

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video", 
                "video": "/home/zhipang/PhysicalDynamics/src/ManualAnnotationSystem/test2.mp4",
            },
            {'type': 'text', 'text': '''describe this video'''},
        ],
    },
]
num_frames = None
fps = None

print(type(model))
print(gap_line())
model.show_tokenized_inputs(conversation=conversation, num_frames=num_frames, fps=fps)
print(gap_line())
# exit()
print(model(conversation=conversation, num_frames=num_frames, fps=fps))
print(gap_line())
