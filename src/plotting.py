"""学习曲线与可比较的实验结果。"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from config import MODEL_TYPES


def plot_history(history):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ("loss", "perplexity", "accuracy"),
                                ("Cross entropy", "Perplexity", "Next-token accuracy")):
        ax.plot(frame["epoch"], frame[f"train_{metric}"], label="train")
        ax.plot(frame["epoch"], frame[f"valid_{metric}"], label="valid")
        ax.set(xlabel="Epoch", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def comparison_table(output_dir):
    """仅合并同一语料、词表和超参数的实验，避免误比较旧结果。"""
    rows = []
    reference = None
    for model_type in MODEL_TYPES:
        path = Path(output_dir) / model_type / "metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        signature = {
            "corpus": metrics["data"]["corpus_sha256"],
            "vocab": metrics["data"]["vocab_sha256"],
            "tokenization": metrics["data"]["tokenization"],
            "config": metrics["experiment_config"],
        }
        if reference is not None and signature != reference:
            raise ValueError("模型的语料、词表或训练参数不同，请使用相同配置重新训练，或指定独立输出目录。")
        reference = signature
        rows.append({
            "model": model_type.upper(),
            "best_epoch": metrics["best_epoch"],
            "parameters": metrics["parameter_count"],
            "valid_loss": metrics["valid"]["loss"],
            "valid_perplexity": metrics["valid"]["perplexity"],
            "test_loss": metrics["test"]["loss"],
            "test_perplexity": metrics["test"]["perplexity"],
            "test_accuracy": metrics["test"]["accuracy"],
            "test_known_accuracy": metrics["test"]["known_accuracy"],
            "test_unknown_target_rate": metrics["test"]["unknown_target_rate"],
            "train_seconds": metrics["train_seconds"],
            "device": metrics["device"],
        })
    return pd.DataFrame(rows).sort_values("valid_perplexity").reset_index(drop=True) if rows else pd.DataFrame()
