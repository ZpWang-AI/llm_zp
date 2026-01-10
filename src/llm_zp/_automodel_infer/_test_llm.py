from utils_zp import *
from llm_zp import Qwen2_5_VL, InternVL3_5, LLaVA_NeXT_Video, ConversationInput_zp, Qwen3

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def test_mllm():
    # model = Qwen2_5_VL('/home/zhipang/pretrained_models/Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
    # model = InternVL3_5('/home/zhipang/pretrained_models/InternVL3_5-8B-HF')
    # model = LLaVA_NeXT_Video('/home/zhipang/pretrained_models/LLaVA-NeXT-Video-7B-hf')
    model = Qwen2_5_VL('/root/autodl-fs/_fs_data/pretrained_models/Qwen2.5-VL-7B-Instruct')

    conversation = ConversationInput_zp()
    conversation.add('system', 'hello world')
    conversation.add('user', video='/root/autodl-fs/_fs_data/PhyDy_data/data/WISA-80K/encoded_video/0/0ba50e8a7dbba5a2ea95d9a75d997eb157968d4c87663b70bfb3b38af8305802.mp4')
    conversation.add('user', text='describe it')
    num_frames = None
    fps = None

    print(gap_line())
    model.show_tokenized_inputs(conversation=conversation, num_frames=num_frames, fps=fps)
    print(gap_line())
    # exit()
    print(model(conversation=conversation, num_frames=num_frames, fps=fps))
    print(gap_line())


def test_llm():
    model = Qwen3('/root/autodl-fs/_fs_data/pretrained_models/Qwen3-8B')
    print(model([
        {"role": "system", "content": "you're a helpful AI bot",},
        {"role": "user", "content": 'how are you',},
    ]))


if __name__ == '__main__':
    # test_mllm()
    # test_llm()

    pass

