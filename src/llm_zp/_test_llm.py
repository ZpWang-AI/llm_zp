from utils_zp import *
from llm_zp import Qwen2_5_VL, InternVL3_5, LLaVA_NeXT_Video

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

model = Qwen2_5_VL()
model = InternVL3_5()
model = LLaVA_NeXT_Video()

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
num_frames = 10
fps = 1

print(type(model))
print(gap_line())
model.show_tokenized_inputs(conversation=conversation, num_frames=num_frames, fps=fps)
print(gap_line())
# exit()
print(model(conversation=conversation, num_frames=num_frames, fps=fps))
print(gap_line())
