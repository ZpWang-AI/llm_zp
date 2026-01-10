from ._api_call_zp import APICalling_zp
from ._api_call import APICalling, APICalling_MultiModal
from ._api_key import api_key_dic


class APIModelName:
    """Constants for API model identifiers. Each attribute is a string representing a specific model name."""
    
    # General-purpose models
    deepseek_chat = 'deepseek-chat'
    
    # Qwen3 text generation models
    # qwen3_0_6b = 'qwen3-0.6b'
    # qwen3_1_7b = 'qwen3-1.7b'
    # qwen3_4b = 'qwen3-4b'
    # qwen3_8b = 'qwen3-8b'
    # qwen3_14b = 'qwen3-14b'
    qwen3_32b = 'qwen3-32b'
    qwen3_30b_a3b = 'qwen3-30b-a3b'
    qwen3_235b_a22b = 'qwen3-235b-a22b'
    
    # Qwen3 enhanced models
    qwen3_max = 'qwen3-max'
    qwen3_coder_plus = 'qwen3-coder-plus'
    
    # Qwen3 multimodal models
    qwen3_vl_flash = 'qwen3-vl-flash'
    qwen3_vl_plus = 'qwen3-vl-plus'
    
    # Qwen3 retrieval models
    qwen3_rerank = 'qwen3-rerank'
    
    # Embedding models
    text_embedding_v4 = 'text-embedding-v4'


class APIURLName:
    """Constants for API base URLs. Each attribute provides the base endpoint URL for a service provider."""
    
    deepseek = "https://api.deepseek.com"
    openai = "https://api.openai.com"
    anthropic = "https://api.anthropic.com"


