from utils_zp import *


FuncParams = ParamSpec("FuncParams")
ReturnType = TypeVar("ReturnType")
GenericFunc = Callable[FuncParams, ReturnType]


@dataclass
class APICalling_zp:
    api_key: str = ''
    model: str = ''
    base_url: str = ''
    record_dir: Union[Literal['tmp'], None, str, path] = 'tmp'
    print_io: bool = False
    err_jsonl: Union[Literal['tmp'], None, str, path] = None
    err_output: str = ''
    raise_err: bool = True

    def __post_init__(self):
        if self.record_dir == 'tmp':
            self.record_dir = path(__file__).parent / '~tmp' / f'{self.model}'
        if self.err_jsonl == 'tmp':
            self.err_jsonl = path(__file__).parent / '~tmp' / f'{self.model}.jsonl'

    def _record_decorator(self, func:GenericFunc) -> GenericFunc:
        if self.record_dir is None: return func
        _record_dic = FileDict(self.record_dir)
        def new_func(messages, *args, **kwargs):
            k = f'{self.model} | {messages} | {args} | {kwargs}'
            v = _record_dic[k]
            if v is not None: return v
            _ret = func(messages, *args, **kwargs)
            _record_dic[k] = _ret
            return _ret
        return new_func

    def _print_io_decorator(self, func:GenericFunc) -> GenericFunc:
        if not self.print_io: return func
        def new_func(messages, *args, **kwargs):
            print(messages)
            if args: print(args)
            if kwargs: print(kwargs)
            print(gap_line(fillchar='-'))
            _ret = func(messages, *args, **kwargs)
            print(_ret)
            print(gap_line())
            return _ret
        return new_func

    def _err_decorator(self, func:GenericFunc) -> GenericFunc:
        def new_func(messages, *args, **kwargs):
            try:
                _ret = func(messages, *args, **kwargs)
                return _ret
            except Exception as err:
                if self.err_jsonl:
                    auto_dump({'time':str(datetime.datetime.now()), 'err':str(err), 'query':messages, 'traceback': traceback.format_exc()}, self.err_jsonl)
                if self.raise_err: raise err
                return self.err_output
        return new_func
    
    def _decorator(self, func:GenericFunc) -> GenericFunc:
        decorator_list = [
            self._err_decorator,
            self._print_io_decorator,
            self._record_decorator,
        ]
        for d in decorator_list: func = d(func)
        return func

    @property
    def openai_client(self): 
        import openai
        return openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat_openai(self, query:str=None, messages:list=None,) -> str:
        """
~~~
[{'role': 'user', 'content': query}]
~~~
        """
        if query is not None:
            assert messages is None
            messages = [{'role': 'user', 'content': query}]
        else:
            assert messages is not None

        def func(messages):
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            answer = response.choices[0].message.content
            return answer

        return self._decorator(func)(messages)
    
    def embed_openai(self, query:str, dimensions:int) -> List[float]:
        def func(query:str, dimensions:int):
            response = self.openai_client.embeddings.create(
                model=self.model,
                input=query,
                dimensions=dimensions,
            )
            return response.data[0].embedding
        
        return self._decorator(func)(query, dimensions=dimensions)

    def multimodal_chat_dashscope(self, messages:list) -> str:
        """
~~~
[
    {
        "role": "user",
        "content": [
            {'video': 'file:///root/autodl-fs/_fs_data/PhyDy_data/data/WISA-80K/video/0/0ba50e8a7dbba5a2ea95d9a75d997eb157968d4c87663b70bfb3b38af8305802.mp4'},
            {"text": "first, describe this vidoe."},
            {"image": "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg"},
            {"text": "then, solve this problem"},
        ]
    }
]
~~~
        """
        def func(messages:list):
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

        return self._decorator(func)(messages)

    def rerank_dashscope(
        self, query:str, documents:List[str],
        top_n:int=10, 
        instruct="Given a query, retrieve relevant documents about the query.",
        return_documents=True,
    ) -> Tuple[List[str], List[float]]:
        import dashscope
        dashscope.api_key = self.api_key
        def func(query, documents, top_n, instruct):
            resp = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=return_documents,
                instruct=instruct
            )
            sorted_documents = []
            relevance_scores = []
            try:
                for case in resp.output.results:
                    sorted_documents.append(case['document']['text'])
                    relevance_scores.append(case.relevance_score)
            except Exception as e:
                print(traceback.format_exc())
                print(gap_line())
                print(resp)
                print(gap_line())
                print('fail to get response from api')
                exit()
            return sorted_documents, relevance_scores
        return self._decorator(func)(query,documents,top_n,instruct)

