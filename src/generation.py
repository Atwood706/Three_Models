"""加载模型、预测下一个单词，以及自回归生成文本。"""
import math

import torch

from src.model_factory import create_model_from_config
from src.preprocessing import numericalize, tokenize


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = create_model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint["vocab"], checkpoint


def _prefix_tensor(prefix, vocab, device):
    tokens = tokenize(prefix)
    if not tokens:
        raise ValueError("prefix 必须至少包含一个英文单词。")
    X = torch.tensor([numericalize(tokens, vocab)], dtype=torch.long, device=device)
    return tokens, X


def _ordinary_word_logits(logits, vocab):
    logits = logits.clone()
    # <pad> 和 <unk> 不是可展示的单词，生成时屏蔽并重新归一化。
    logits[vocab["<pad>"]] = -float("inf")
    logits[vocab["<unk>"]] = -float("inf")
    return logits


@torch.no_grad()
def predict_next(model, vocab, prefix, top_k=5):
    if top_k < 1:
        raise ValueError("top_k 必须大于等于 1。")
    if len(vocab) <= 2:
        raise ValueError("词表必须包含普通单词。")
    device = next(model.parameters()).device
    _, X = _prefix_tensor(prefix, vocab, device)
    was_training = model.training
    model.eval()
    try:
        logits, _ = model(X)
        probabilities = torch.softmax(_ordinary_word_logits(logits[0, -1], vocab), dim=-1)
        values, indices = torch.topk(probabilities, min(top_k, len(vocab) - 2))
        index_to_token = {index: token for token, index in vocab.items()}
        return [
            {"token": index_to_token[index], "probability": probability}
            for index, probability in zip(indices.cpu().tolist(), values.cpu().tolist())
        ]
    finally:
        model.train(was_training)


@torch.no_grad()
def generate_text(model, vocab, prefix, num_words=50, temperature=0.8, top_k=20, seed=42):
    """temperature=0 使用 argmax；top_k=0 从所有普通单词中采样。"""
    if num_words < 0 or top_k < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("num_words、top_k、temperature 必须为非负数，temperature 必须有限。")
    if len(vocab) <= 2:
        raise ValueError("词表必须包含普通单词。")
    device = next(model.parameters()).device
    tokens, X = _prefix_tensor(prefix, vocab, device)
    index_to_token = {index: token for token, index in vocab.items()}
    # 在 CPU 采样，从而兼容 MPS，并且不改动训练使用的全局 RNG。
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        if num_words == 0:
            return " ".join(tokens)
        logits, state = model(X)
        for step in range(num_words):
            scores = _ordinary_word_logits(logits[0, -1], vocab)
            if temperature == 0:
                next_id = int(scores.argmax().item())
            else:
                scores = scores / temperature
                if top_k:
                    values, indices = scores.topk(min(top_k, len(vocab) - 2))
                    filtered = torch.full_like(scores, -float("inf"))
                    filtered[indices] = values
                    scores = filtered
                probabilities = torch.softmax(scores, dim=-1).cpu()
                next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
            tokens.append(index_to_token[next_id])
            if step + 1 < num_words:
                X = torch.tensor([[next_id]], dtype=torch.long, device=device)
                logits, state = model(X, state)
        return " ".join(tokens)
    finally:
        model.train(was_training)
