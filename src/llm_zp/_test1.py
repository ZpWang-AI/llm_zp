from utils_zp import *
from llm_zp import QwenVL

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

model = QwenVL('/home/zhipang/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/checkpoint-3')

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
                # "video": "file:///home/zhipang/PhysicalDynamics/data/Annotation/pipeline.wisa_v3_2.example/yes/3/1d9c24ec6d0d48b33ef4765edc267a483993eada518649a29620c5861f9bfe13.mp4",
                "video": "file:///home/zhipang/PhysicalDynamics/src/ManualAnnotationSystem/test2.mp4",
                "max_pixels": 128 * 128,
                "fps": 15.0,
            },
            {'type': 'text', 'text': '''## Objective
Identify the dominant subjects in the video.

## Guidelines
* List subjects with concise noun phrases, one per line.
* Prioritize subjects that are active, dynamic, or central to the scene.
* Ignore static/background objects (e.g., trees, furniture) and minor/secondary elements.
* If there are no dominant subjects, simply output "None".

## Output Examples
### Example 1
woman in red dress
golden pet dog

### Example 2
burning campfire
ice cube

### Example 3
None'''}
        ],
    },
]
print()
print(
model(conversation)[0]

)