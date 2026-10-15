"""正文提取、单词分词、按顺序切分、训练词表与数值化。"""
from collections import Counter
from pathlib import Path
import re


def extract_body(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*", text, re.M)
    if start:
        text = text[start.end():]
    end = re.search(r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*", text, re.M)
    if end:
        text = text[:end.start()]
    # 本书的第一章标题：去掉书名、作者和目录，但保留后续章节的正文顺序。
    introduction = re.search(r"^[ \t]*I\.[ \t]*\n[ \t]*Introduction[ \t]*\n", text, re.M)
    if introduction:
        text = text[introduction.end():]
    return text.strip()


def read_text(path):
    return extract_body(Path(path).read_text(encoding="utf-8-sig"))


def tokenize(text):
    """一个 token 是一个英文单词；don't / traveller's 保持为一个 token。"""
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text)


def clean_text(text):
    return " ".join(tokenize(text))


def split_tokens(tokens, train_ratio=0.8, valid_ratio=0.1):
    if not (0 < train_ratio < 1 and 0 < valid_ratio < 1 and train_ratio + valid_ratio < 1):
        raise ValueError("训练、验证、测试集比例必须均大于 0。")
    train_end = int(len(tokens) * train_ratio)
    valid_end = int(len(tokens) * (train_ratio + valid_ratio))
    return {
        "train": tokens[:train_end],
        "valid": tokens[train_end:valid_end],
        "test": tokens[valid_end:],
    }


def build_vocab(tokenized_texts, min_freq=2):
    """与旧项目一致的接口：输入多个 token 列表，只用训练数据构建。"""
    if min_freq < 1:
        raise ValueError("min_freq 必须大于等于 1。")
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def numericalize(tokens, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def decode(indices, vocab):
    index_to_token = {index: token for token, index in vocab.items()}
    return [index_to_token[int(index)] for index in indices]


def unknown_rate(tokens, vocab):
    return sum(token not in vocab for token in tokens) / len(tokens) if tokens else 0.0
