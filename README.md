# LLM_API

Use LLM by API or local model.

## Quick Start

1. installation

~~~sh
pip install -r requirements.txt
~~~

2. usage

~~~python
from llm_zp import Qwen2_5_VL
Qwen2_5_VL = Qwen2_5_VL('Qwen/Qwen2.5-7B-Instruct')
query = [
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
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
                "max_pixels": 128 * 128,
                "fps": 15.0,
            },
            {'type': 'text', 'text': 'hello world'}
        ],
    },
]
answer = Qwen2_5_VL(query)
print(answer)
~~~

~~~python
from llm_zp import APICalling
api = APICalling(
    api_key='xxx',
    base_url='',
    model='',
)
query = 'hello world'
answer = api(query)
print(answer)
~~~



<!-- 3. `llm_api()
1. edit `src/template_script/001-pdtb3_top_subtext.py`

2. run `sh run.sh 001`

3. run `python src/process_pred.py` -->
