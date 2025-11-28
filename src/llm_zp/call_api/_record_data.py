from utils_zp import *

from hashlib import blake2b

class RecordDatabase:
    def __init__(self, data_dir, hash_space = 10**5, hash_stack_size = 100):
        import math
        self.data_dir = path(data_dir)
        make_path(self.data_dir)

        self.digest_size = int(math.log(hash_space/hash_stack_size, 16))
        self.digest_size = max(1, self.digest_size)

    def _hash(self, s:str) -> str:
        return blake2b(str(s).encode(), digest_size=self.digest_size).hexdigest()

    def __contains__(self, key:str) -> bool:
        _hash = self._hash(key)
        filepath = self.data_dir / f'{_hash}.json'
        if filepath.exists():
            _dic = auto_load(filepath)
            return key in _dic
        
    def __getitem__(self, key:str) -> Optional[str]:
        _hash = self._hash(key)
        filepath = self.data_dir / f'{_hash}.json'
        if filepath.exists():
            _dic = auto_load(filepath)
            return _dic.get(key, None)
        
    def __setitem__(self, key:str, val:str):
        _hash = self._hash(key)
        filepath = self.data_dir / f'{_hash}.json'
        if filepath.exists():
            _dic = auto_load(filepath)
            _dic[key] = val
        else:
            _dic = {key:val}
        auto_dump(_dic, filepath)

    def clear(self):
        shutil.rmtree(self.data_dir)
        make_path(self.data_dir)


if __name__ == '__main__':
    rdb_dir = path(__file__).parent/'~test_rdb'
    rdb = RecordDatabase(rdb_dir,)
    rdb.clear()
    exit()
    for i in tqdm.tqdm(range(100000)):
        d = random.randint(0,100000)
        if random.random() < 0.6:
            rdb[d] = d
        else:
            val = rdb[d]
            assert val == d or val is None

        

