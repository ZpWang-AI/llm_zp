from utils_zp import *


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
        raise_err=True,
    ):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key, base_url=base_url
        )
        self.model = model

        local_dir = path(__file__).parent / '~tmp'
        self.error_jsonl = error_jsonl if error_jsonl is not None else local_dir / 'err.jsonl'
        print(f'> err file: {self.error_jsonl}')
        self.do_record = do_record
        if do_record:
            if record_dir is None: record_dir = local_dir / str(model)
            print(f'> record dir: {record_dir}')
            self.record_db = FileDict(record_dir)
        self.print_input_output = print_input_output
        self.raise_err = raise_err
    
    def __chat(self, messages:list):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
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
        query:str=None,
        messages:list=None,
        do_embed=False,
        embed_dimensions=2048,
        err_output='',
    ) -> Union[str, List[float]]:
        import openai
        
        assert query or messages
        assert not (query and messages)
        if query:
            messages = [{'role': 'user', 'content': query}]

        if self.print_input_output:
            print(messages)
            print(gap_line(fillchar='-'))

        ans = None
        if self.do_record:
            ans = self.record_db[str(messages)]
        
        if not ans:
            try:
                if do_embed:
                    ans = self.__embed(messages, dimensions=embed_dimensions)
                else:
                    ans = self.__chat(messages)

                if self.do_record:
                    self.record_db[str(messages)] = ans

            except openai.BadRequestError as err:
                auto_dump({'err':str(err), 'query':messages}, self.error_jsonl)
                if self.raise_err: raise err
                ans = err_output
        
        if self.print_input_output:
            print(ans)
            print(gap_line())

        return ans


class APICalling_MultiModal:
    def __init__(
        self, 
        api_key,
        model, 
        error_jsonl=None,
        do_record=False,
        record_dir=None, 
        print_input_output=False,
        raise_err=True,
    ):
        self.api_key = api_key
        self.model = model

        local_dir = path(__file__).parent / '~tmp'
        self.error_jsonl = error_jsonl if error_jsonl is not None else local_dir / 'err.jsonl'
        print(f'> API err file: {self.error_jsonl}')
        self.do_record = do_record
        if do_record:
            if record_dir is None: record_dir = local_dir / str(model)
            print(f'> API record dir: {record_dir}')
            self.record_db = FileDict(record_dir)
        self.print_input_output = print_input_output
        self.raise_err = raise_err

    def __chat(self, messages:list):
        from dashscope import MultiModalConversation
        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,  
            messages=messages)
        response = dict(response)
        # print(response); exit()
        # answer = response['output']['choices'][0]['message']['content']
        answer = response['output']['choices'][0]['message']['content'][0]['text']
        return answer

    def __call__(
        self,
        video:str=None,
        query:str=None,
        messages:list=None,
        fps:float=2.0,
        err_output='',
    ) -> Union[str, List[float]]:
        if video or query:
            assert (video and query) and not messages
            messages = [
                {'role':'user',
                'content': [{'video': video,"fps":fps},
                            {'text': query}]}]
        elif messages:
            assert not video and not query

        if self.print_input_output:
            print(messages)
            print(gap_line(fillchar='-'))

        ans = None
        if self.do_record:
            ans = self.record_db[str(messages)]
        
        if not ans:
            try:
                ans = self.__chat(messages)

                if self.do_record:
                    self.record_db[str(messages)] = ans

            except Exception as err:
                auto_dump({'err':str(err), 'query':messages}, self.error_jsonl)
                if self.raise_err: raise err
                ans = err_output
        
        if self.print_input_output:
            print(ans)
            print(gap_line())

        return ans

