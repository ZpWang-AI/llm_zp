from utils_zp import *


class LLMBaseClass_zp:
    def __init__(
        self,
        model_or_model_path,
        max_new_tokens=1024,
        mode:Literal['bf16']='bf16', 
        input_device:Literal['auto', 'cuda:0', 'cuda:1']='auto',

        batch_output=False
    ):
        self._model = None
        self._processor = None
        self.model_or_model_path = model_or_model_path
        self.max_new_tokens = max_new_tokens
        self.model_load_mode = mode
        self.input_device = input_device

        self.batch_output = batch_output

    @property
    def model(self): raise Exception('todo model')
    
    @property
    def processor(self): 
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.model_or_model_path,
                use_fast=True,
            )
        return self._processor

    def tokenize(self, conversation, fps:float=None, num_frames:int=None):
        if fps is not None:
            max_num_frames = -1
            for qa in conversation:
                for content in qa['content']:
                    if content['type'] == 'video':
                        cur_num_frames = int(Video_custom(content['video']).duration*fps)
                        max_num_frames = max(max_num_frames, cur_num_frames)
            num_frames = max_num_frames if max_num_frames > 0 else None

        inputs = self.processor.apply_chat_template(
            conversation,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
        )
        return inputs
    
    def show_tokenized_inputs(self, conversation, fps:float=None, num_frames:int=None):
        inputs = self.tokenize(conversation, num_frames=num_frames, fps=fps)
        for k in inputs:
            print(k, ':', inputs[k].shape)

    def __call__(self, conversation, fps:float=None, num_frames:int=None):
        inputs = self.tokenize(conversation, num_frames=num_frames, fps=fps)
        if self.input_device == 'auto':
            inputs = inputs.to(self.model.device).to(self.model.dtype)
        else:
            inputs = inputs.to(self.input_device).to(self.model.dtype)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
        torch.cuda.empty_cache()

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text:List[str] = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if self.batch_output:
            return output_text
        else:
            assert len(output_text)==1
            return output_text[0]
        