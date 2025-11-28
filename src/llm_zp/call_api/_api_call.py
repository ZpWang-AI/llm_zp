from utils_zp import *

import openai
from openai import OpenAI

from ._record_data import RecordDatabase


class APICalling:
    def __init__(
        self, 
        api_key, 
        base_url, 
        model, 
        error_jsonl=None,
        do_record=False,
        record_dir=None, 
        print_input_output=False,
    ):
        local_dir = path(__file__).parent / '~tmp'
        self.client = OpenAI(
            api_key=api_key, base_url=base_url
        )
        self.model = model

        self.error_jsonl = error_jsonl if error_jsonl is not None else local_dir / 'err.jsonl'
        self.do_record = do_record
        if do_record:
            self.record = RecordDatabase(record_dir if record_dir is not None else local_dir / str(model))
        self.print_input_output = print_input_output

    
    def __pure_call(self, query: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'user', 'content': query}
            ],
            stream=False
        )
        answer = response.choices[0].message.content
        return answer
    
    def __call__(self, query:str):
        if self.print_input_output:
            print(query)
            print(gap_line(fillchar='-'))

        if self.do_record and query in self.record:
            ans = self.record[query]
        else:
            try:
                ans = self.__pure_call(query)
            except openai.BadRequestError as err:
                auto_dump({'err':str(err), 'query':query}, self.error_jsonl)
                ans = ''

        if self.do_record:
            self.record[query] = ans
        
        if self.print_input_output:
            print(ans)
            print(gap_line())

        return ans