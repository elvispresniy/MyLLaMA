import pickle
import regex as re

from config import CUSTOM_SPLIT_PATTERN

class Tokenizer:
    def __init__(self):
        self.pair2idx = {}
        self.idx2bytes = {idx:bytes([idx]) for idx in range(256)}
        self.compiled_pattern = re.compile(CUSTOM_SPLIT_PATTERN)


    def _get_stats(self, ids_chunks: list[list[int]]):
        vocab = {}
        for ids in ids_chunks:
            for pair in zip(ids, ids[1:]):
                vocab[pair] = vocab.get(pair, 0) + 1
        return vocab


    def _merge(self, ids_chunks: list[list[int]], pair: tuple[int, int], idx: int):
        for ids in ids_chunks:
            i = 0
            while i < len(ids) - 1:
                if (ids[i], ids[i + 1]) == pair:
                    ids[i] = idx
                    del ids[i + 1]
                i += 1

        return ids_chunks


    def train(self, texts: list[str], max_vocab_size: int):
        assert max_vocab_size >= 256
        num_new_pairs = max_vocab_size - 256

        tmp_text_chunks = [re.findall(self.compiled_pattern, text) for text in texts]
        text_chunks = []
        for chunk in tmp_text_chunks:
            text_chunks.extend(chunk)

        ids_chunks = [list(ch.encode("utf-8")) for ch in text_chunks]

        for i in range(num_new_pairs):
            vocab = self._get_stats(ids_chunks)
            most_common_pair = max(vocab, key=vocab.get)
            idx = 256 + i
            self.pair2idx[most_common_pair] = idx
            ids_chunks = self._merge(ids_chunks, most_common_pair, idx)

        for (p0, p1), idx in self.pair2idx.items():
            self.idx2bytes[idx] = self.idx2bytes[p0] + self.idx2bytes[p1]


    def decode(self, ids: list[int]):
        tokens = b"".join(self.idx2bytes[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


    def encode(self, text: str):
        ids = list(text.encode("utf-8"))

        while True:
            i = 0
            flag = False
            while i < len(ids) - 1:
                pair = (ids[i], ids[i + 1])
                if pair in self.pair2idx:
                    ids[i] = self.pair2idx[pair]
                    del ids[i + 1]
                    flag = True
                i += 1
            if flag == False:
                break

        return ids
    
    @classmethod
    def from_pickle_files(cls, pair2idx_path, idx2bytes_path):
        tokenizer = cls()
        with open(pair2idx_path, 'rb') as f:
            tokenizer.pair2idx = pickle.load(f)
        with open(idx2bytes_path, 'rb') as f:
            tokenizer.idx2bytes = pickle.load(f)
        return tokenizer