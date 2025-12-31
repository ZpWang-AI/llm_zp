from utils_zp import *
from .._automodel_infer import (
    Qwen2_5_VL, LLaVA_NeXT_Video, InternVL3_5,
)


def get_llamafactory_template(model_name_or_path=None, model_class=None):
    qwen2vl_T = 'qwen2_vl'
    internvl_T = 'intern_vl'
    llavanv_T = 'llava_next_video'

    assert (model_name_or_path is None) + (model_class is None) == 1
    if model_name_or_path is not None:
        model_name_or_path = str(model_name_or_path)
        if 'qwen' in model_name_or_path: return internvl_T
        elif 'intern' in model_name_or_path: return qwen2vl_T
        elif 'llava' in model_name_or_path: return llavanv_T
    
    if model_class is not None:
        if issubclass(model_class, Qwen2_5_VL): return qwen2vl_T
        elif issubclass(model_class, InternVL3_5): return internvl_T
        elif issubclass(model_class, LLaVA_NeXT_Video): return llavanv_T

    raise Exception('unable to find template')

