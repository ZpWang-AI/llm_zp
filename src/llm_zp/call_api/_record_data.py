from utils_zp import *

from hashlib import blake2b

class RecordDatabase:
    def __init__(self, data_dir, digest_size=2):
        # max database size = 256**digest_size
        self.data_dir = path(data_dir)
        make_path(self.data_dir)

        assert digest_size >= 1
        self.digest_size = digest_size

    def _hash(self, s) -> str:
        return blake2b(str(s).encode(), digest_size=self.digest_size).hexdigest()

    def __iter__(self):
        def _func():
            for _file in oswalk_full_path(self.data_dir, only_file=True):
                _dic:dict = auto_load(_file)
                for k,v in _dic.items():
                    yield k,v
        return _func()
    
    @classmethod
    def __check_key(cls, key:str) -> bool:
        return isinstance(key, (str,int,float,tuple))
    
    @classmethod
    def __dump(cls, filepath, _dic):
        with open(filepath, 'w', encoding=utf8)as f:
            json.dump(_dic, f, ensure_ascii=False)

    def __contains__(self, key) -> bool:
        _hash = self._hash(key)
        filepath = self.data_dir / f'{_hash}.json'
        if filepath.exists():
            _dic = auto_load(filepath)
            return key in _dic
        
    def __getitem__(self, key) -> Optional[str]:
        _hash = self._hash(key)
        filepath = self.data_dir / f'{_hash}.json'
        if filepath.exists():
            _dic = auto_load(filepath)
            return _dic.get(key, None)
    
    def __delitem__(self, key):
        _hash = self._hash(key)
        filepath = self.data_dir / f'{_hash}.json'
        if filepath.exists():
            _dic = auto_load(filepath)
            if key in _dic: 
                del _dic[key]
                self.__dump(filepath, _dic)
        
    def __setitem__(self, key, val):
        _hash = self._hash(key)
        filepath = self.data_dir / f'{_hash}.json'
        if filepath.exists():
            _dic = auto_load(filepath)
            _dic[key] = val
        else:
            _dic = {key:val}
        self.__dump(filepath, _dic)
        # auto_dump(_dic, filepath)

    def clear(self):
        shutil.rmtree(self.data_dir)
        make_path(self.data_dir)


if __name__ == '__main__':
    rdb_dir = path(__file__).parent/'~test_rdb'
    shutil.rmtree(rdb_dir); exit()
    rdb = RecordDatabase(rdb_dir,)
    rdb.clear()
    for i in tqdm.tqdm(range(5000)):
        d = random.randint(0,20)
        if random.random() < 0.6:
            rdb[d] = d
        else:
            val = rdb[d]
            assert val == d or val is None
    print(sorted(rdb))
        

