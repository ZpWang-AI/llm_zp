from utils_zp import *

import openai
from openai import OpenAI


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
        self.client = OpenAI(
            api_key=api_key, base_url=base_url
        )
        self.model = model

        local_dir = path(__file__).parent / '~tmp'
        self.error_jsonl = error_jsonl if error_jsonl is not None else local_dir / 'err.jsonl'
        self.do_record = do_record
        if do_record:
            if record_dir is None: record_dir = local_dir / str(model)
            self.record_db = FileDict(record_dir)
        self.print_input_output = print_input_output
    
    def __chat(self, query:str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'user', 'content': query}
            ],
            stream=False
        )
        answer = response.choices[0].message.content
        return answer
    
    def __embed(self, query:str, dimensions:int) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=query,
            dimensions=dimensions,
        )
        return response.data[0].embedding

    def __call__(
        self,
        query:str,
        do_embed=False,
        embed_dimensions=2048,
        err_output='',
    ) -> Union[str, List[float]]:
        if self.print_input_output:
            print(query)
            print(gap_line(fillchar='-'))

        ans = None
        if self.do_record:
            ans = self.record_db[query]
        
        if not ans:
            try:
                if do_embed:
                    ans = self.__embed(query, dimensions=embed_dimensions)
                else:
                    ans = self.__chat(query)

                if self.do_record:
                    self.record_db[query] = ans

            except openai.BadRequestError as err:
                auto_dump({'err':str(err), 'query':query}, self.error_jsonl)
                ans = err_output
        
        if self.print_input_output:
            print(ans)
            print(gap_line())

        return ans
    