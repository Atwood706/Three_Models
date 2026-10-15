"""共享实验参数与 Mac / Kaggle / 服务器路径、设备设置。"""
from dataclasses import dataclass
from pathlib import Path
import os

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_TYPES = ("rnn", "lstm", "gru")


def resolve_data_path(data_path=None):
    """显式路径优先；本地读取原文件；Kaggle 可从挂载数据集读取。"""
    explicit = data_path or os.environ.get("TIME_MACHINE_DATA_PATH")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"文本不存在: {path}")
        return path

    local_path = PROJECT_ROOT / "pg35.txt"
    if local_path.is_file():
        return local_path

    kaggle_input = Path("/kaggle/input")
    candidates = sorted(kaggle_input.rglob("pg35.txt")) if kaggle_input.exists() else []
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("找到多个 pg35.txt，请用 --data-path 或 TIME_MACHINE_DATA_PATH 指定。")
    raise FileNotFoundError("找不到 pg35.txt，请用 --data-path 或 TIME_MACHINE_DATA_PATH 指定。")


def resolve_output_dir(output_dir=None):
    explicit = output_dir or os.environ.get("TIME_MACHINE_OUTPUT_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/time_machine")
    return PROJECT_ROOT / "outputs"


def get_device(requested="auto"):
    mps = getattr(torch.backends, "mps", None)
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": mps is not None and mps.is_available(),
    }
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available:
        raise ValueError("device 必须为 auto、cuda、mps 或 cpu。")
    if not available[requested]:
        raise RuntimeError(f"当前环境不支持 {requested}，可指定 --device cpu。")
    return torch.device(requested)


@dataclass
class ExperimentConfig:
    seed: int = 42
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    min_freq: int = 2
    seq_len: int = 32
    batch_size: int = 32
    embed_size: int = 100
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 0.001
    epochs: int = 20
    grad_clip: float = 1.0

    def __post_init__(self):
        for name in ("min_freq", "seq_len", "batch_size", "embed_size", "hidden_size", "num_layers", "epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须大于等于 1。")
        if not (0 < self.train_ratio < 1 and 0 < self.valid_ratio < 1
                and self.train_ratio + self.valid_ratio < 1):
            raise ValueError("训练、验证、测试集比例必须均大于 0。")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1)。")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate 和 grad_clip 必须大于 0。")
