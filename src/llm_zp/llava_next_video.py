from utils_zp import *

"""
https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf

"""


class LLaVA_NeXT_Video:
    def __init__(
        self, 
        model_or_model_path='/home/zhipang/pretrained_models/LLaVA-NeXT-Video-7B-hf', 
        # mode:Literal['auto', 'bf16', '4bit', '8bit']='bf16', 
        mode:Literal['bf16']='bf16', 
        input_device:Literal['auto', 'cuda:0', 'cuda:1']='auto',
        max_new_tokens=1024,
    ):
        # if mode == 'auto':
        #     model_arg_map = {
        #         'torch_dtype': 'auto',
        #         'device_map': 'auto',
        #     }
        if mode == 'bf16':
            model_arg_map_lambda = lambda: {
                'torch_dtype': torch.bfloat16,
                'device_map': 'cuda',
                # 'device_map': 'auto',
                # 'attn_implementation': 'flash_attention_2'
            }
        # elif mode == '4bit':
        #     bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
        #     model_arg_map = {
        #         'torch_dtype': torch.bfloat16,
        #         'device_map': 'auto',
        #         'quantization_config': bnb_config,
        #         'attn_implementation': 'flash_attention_2'
        #     }
        # elif mode == '8bit':
        #     bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        #     model_arg_map = {
        #         'torch_dtype': torch.bfloat16,
        #         'device_map': 'auto',
        #         'quantization_config': bnb_config,
        #         'attn_implementation': 'flash_attention_2'
        #     }
        else:
            raise Exception(f'wrong mode: {mode}')

        self.processor = None
        self.model = None
        self.model_or_model_path = model_or_model_path
        self.model_arg_map_lambda = model_arg_map_lambda
        self.input_device = input_device
        self.max_new_tokens = max_new_tokens
    
    def load_model(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if self.model is None:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_or_model_path, 
                **(self.model_arg_map_lambda())
            )
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_or_model_path,
                use_fast=True,
            )

    def __call__(self, conversation, num_frames:int=20) -> str:
        from transformers import LlavaNextVideoProcessor

        if not self.model or not self.processor:
            self.load_model()

        model, processor, input_device = self.model, self.processor, self.input_device
        
        processor:LlavaNextVideoProcessor
        inputs = processor.apply_chat_template(
            conversation,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
            # fps=fps,
        )

        if input_device == 'auto':
            inputs = inputs.to(model.device).to(model.dtype)
        else:
            inputs = inputs.to(input_device).to(model.dtype)
        assert inputs['input_ids'].shape[0] == 1

        # print(inputs)
        generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        decoded_output = processor.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return decoded_output

        # generated_ids_trimmed = [
        #     out_ids[in_ids.shape[1] :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        # ]
        # output_text:List[str] = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )

        # # if only_output_assistant:
        # #     text = [p.split('assistant\n', 1)[1] for p in text]
        # return output_text


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    conversation = [
        # {
        #     "role": "system",
        #     "content": [
        #         {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        #     ],
        # },
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    # "video": "file:///home/zhipang/PhysicalDynamics/data/Annotation/pipeline.wisa_v3_2.example/yes/3/1d9c24ec6d0d48b33ef4765edc267a483993eada518649a29620c5861f9bfe13.mp4",
                    "video": "/home/zhipang/PhysicalDynamics/src/ManualAnnotationSystem/test2.mp4",
                    # "max_pixels": 128 * 128,
                    # "fps": 15.0,
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
        LLaVA_NeXT_Video()(conversation)

    )