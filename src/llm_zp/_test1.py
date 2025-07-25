from utils_zp import *
from llm_zp import QwenVL
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

# model = QwenVL('/home/zhipang/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/checkpoint-3')
model = QwenVL('/home/zhipang/pretrained_models/Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')

video_content = {
    "type": "video", 
    # "video": "file:///home/zhipang/PhysicalDynamics/data/Annotation/pipeline.wisa_v3_2.example/yes/3/1d9c24ec6d0d48b33ef4765edc267a483993eada518649a29620c5861f9bfe13.mp4",
    "video": "file:///home/zhipang/PhysicalDynamics/src/ManualAnnotationSystem/test2.mp4",
    "max_pixels": 128 * 128,
    "fps": 15.0,
}
video_content2 = {
    "type": "video", 
    # "video": "file:///home/zhipang/PhysicalDynamics/data/Annotation/pipeline.wisa_v3_2.example/yes/3/1d9c24ec6d0d48b33ef4765edc267a483993eada518649a29620c5861f9bfe13.mp4",
    "video": "file:///home/zhipang/PhysicalDynamics/src/ManualAnnotationSystem/test_2.mp4",
    "max_pixels": 128 * 128,
    "fps": 15.0,
}

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
            video_content2,
            {'type': 'text', 'text': '''## Objective
Identify the dominant subjects in the video.

## Guidelines
* List subjects with concise noun phrases, one per line.
* Prioritize subjects that are active, dynamic, or central to the scene.
* Ignore static/background objects (e.g., trees, furniture) and minor/secondary elements.
* If there are no dominant subjects, simply output "None".

## Output Examples
### Example 1
'''},
            video_content,
            {'type': 'text', 'text': '''
woman in red dress
golden pet dog

### Example 2
'''},
            video_content,
            {'type': 'text', 'text': '''
burning campfire
ice cube

### Example 3
'''},
            video_content,
            {'type': 'text', 'text': '''
None'''}
        ],
    },
]



# conversation = [
#     {
#         "role": "system",
#         "content": [
#             {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
#         ],
#     },
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video", 
#                 # "video": "file:///home/zhipang/PhysicalDynamics/data/Annotation/pipeline.wisa_v3_2.example/yes/3/1d9c24ec6d0d48b33ef4765edc267a483993eada518649a29620c5861f9bfe13.mp4",
#                 "video": "file:///home/zhipang/PhysicalDynamics/src/ManualAnnotationSystem/test2.mp4",
#                 "max_pixels": 128 * 128,
#                 "fps": 15.0,
#             },
#             {'type': 'text', 'text': '''## Objective
# Identify the dominant subjects in the video.

# ## Guidelines
# * List subjects with concise noun phrases, one per line.
# * Prioritize subjects that are active, dynamic, or central to the scene.
# * Ignore static/background objects (e.g., trees, furniture) and minor/secondary elements.
# * If there are no dominant subjects, simply output "None".

# ## Output Examples
# ### Example 1
# woman in red dress
# golden pet dog

# ### Example 2
# burning campfire
# ice cube

# ### Example 3
# None'''}
#         ],
#     },
# ]

conversation1 = [
    {
        "role": "user",
        "content": [
            {'type': 'text', 'text': '''hello world'''}
        ],
    },
]
conversation2 = [
    {
        "role": "user",
        "content": [
            {'type': 'text', 'text': '''hello'''},
            {'type': 'text', 'text': ''' world'''},
        ],
    },
]

def get_input_ids(conversation):
    image_inputs, video_inputs = process_vision_info(conversation)
    # print(image_inputs)
    # print(video_inputs)
    # print(gap_line())
    # print([p.shape for p in video_inputs])
    # return
    processor = AutoProcessor.from_pretrained(
        model.model_or_model_path,
        use_fast=True,
    )
    text = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True, 
        tokenize=False
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        fps=15,
    )
    # print(inputs)
    input_ids = inputs['input_ids'].numpy().tolist()
    # print(inputs_id)
    return input_ids

input_ids1 = get_input_ids(conversation1)
auto_dump(input_ids1, path('/home/zhipang/LLM_usage/src/llm_zp')/'~inputs_id1.json')
input_ids2 = get_input_ids(conversation2)
auto_dump(input_ids2, path('/home/zhipang/LLM_usage/src/llm_zp')/'~inputs_id2.json')

print(len(input_ids1[0]),len(input_ids2[0]))
exit()
# return


print()
print(
model(conversation)[0]

)