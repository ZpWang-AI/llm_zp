from utils_zp import *


_src_dir = path(__file__).parent.parent.parent
_api_key_file = _src_dir / '~api_key.yaml'
if not _api_key_file.exists(): auto_dump({'None': None}, _api_key_file)
api_key_dic = auto_load(_api_key_file)

# print(api_keys)
