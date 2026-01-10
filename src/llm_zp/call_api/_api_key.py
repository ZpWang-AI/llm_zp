from utils_zp import *


_api_key_file = path(__file__).parent.parent.parent / '~api_key.yaml'
if not _api_key_file.exists(): auto_dump({'None': None}, _api_key_file)
api_key_dic = auto_load(_api_key_file)

# print(api_keys)
