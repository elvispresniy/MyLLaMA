from model import ModelArgs

args = ModelArgs(
    max_batch_size=32,
    dim=512,
    ffn_hidden_dim=1024,
    max_seq_len=128,
    n_heads=16,
    n_kv_heads=8,
    head_dim=64,
    device='cpu',
    n_layers=16,
    vocab_size=1512
)

text = '''1, 2, 3'''

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
CUSTOM_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)|\p{N}{1,1}| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""