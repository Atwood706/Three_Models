"""先切分语料，再构造 X 与右移一位的 y。"""
from dataclasses import dataclass
import hashlib
import json

import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import build_vocab, numericalize, read_text, split_tokens, tokenize, unknown_rate


class TimeMachineDataset(Dataset):
    def __init__(self, tokens, vocab, seq_len=32):
        if seq_len < 1:
            raise ValueError("seq_len 必须大于等于 1。")
        self.seq_len = seq_len
        self.indices = torch.tensor(numericalize(tokens, vocab), dtype=torch.long)
        if len(self.indices) < seq_len + 1:
            raise ValueError(f"该数据集只有 {len(tokens)} 个 token，至少需要 seq_len + 1 = {seq_len + 1} 个。")

    def __len__(self):
        return (len(self.indices) - 1) // self.seq_len

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start = idx * self.seq_len
        X = self.indices[start:start + self.seq_len]
        y = self.indices[start + 1:start + self.seq_len + 1]
        return X, y


@dataclass
class PreparedData:
    vocab: dict
    tokens: dict
    datasets: dict
    loaders: dict
    metadata: dict


def prepare_data(data_path, config):
    tokens = tokenize(read_text(data_path))
    splits = split_tokens(tokens, config.train_ratio, config.valid_ratio)
    vocab = build_vocab([splits["train"]], min_freq=config.min_freq)
    if len(vocab) <= 2:
        raise ValueError("训练词表没有普通单词，请减小 min_freq 或检查文本。")

    datasets = {name: TimeMachineDataset(part, vocab, config.seq_len) for name, part in splits.items()}
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            generator=torch.Generator().manual_seed(config.seed),
            num_workers=0,
            drop_last=False,
        )
        for name, dataset in datasets.items()
    }
    metadata = {
        "data_path": str(data_path),
        "corpus_sha256": hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest(),
        "vocab_sha256": hashlib.sha256(json.dumps(vocab, sort_keys=True).encode("utf-8")).hexdigest(),
        "tokenization": "lowercase_english_words_with_internal_apostrophes_v1",
        "total_tokens": len(tokens),
        "vocab_size": len(vocab),
        "splits": {
            name: {
                "tokens": len(part),
                "sequences": len(datasets[name]),
                "evaluated_targets": len(datasets[name]) * config.seq_len,
                "unused_tail_targets": len(part) - 1 - len(datasets[name]) * config.seq_len,
                "unknown_rate": unknown_rate(part, vocab),
            }
            for name, part in splits.items()
        },
    }
    return PreparedData(vocab, splits, datasets, loaders, metadata)
