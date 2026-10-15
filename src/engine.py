"""共享训练、验证、保存最佳模型和测试评估逻辑。"""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import math
import random
import time

import torch
from torch import nn

from config import resolve_output_dir
from src.generation import generate_text, load_model


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _run_epoch(model, data_loader, criterion, device, optimizer=None, grad_clip=1.0, unk_idx=1):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_known_correct = total_tokens = total_known_tokens = 0
    with torch.set_grad_enabled(training):
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # 每个 batch 是独立的固定长度窗口，因此这里不传入上个 batch 的隐藏状态。
            logits, _ = model(X)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Loss 出现 NaN/Inf，请检查学习率和数据。")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            predicted = logits.argmax(dim=-1)
            known = y != unk_idx
            count = y.numel()
            total_loss += loss.item() * count
            total_correct += (predicted == y).sum().item()
            total_known_correct += ((predicted == y) & known).sum().item()
            total_known_tokens += known.sum().item()
            total_tokens += count
    if total_tokens == 0:
        raise ValueError("DataLoader 没有可用于计算 Loss 的 token。")
    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "accuracy": total_correct / total_tokens,
        "known_accuracy": total_known_correct / total_known_tokens if total_known_tokens else None,
        "tokens": total_tokens,
        "unknown_target_rate": 1 - total_known_tokens / total_tokens,
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
    return _run_epoch(model, train_loader, criterion, device, optimizer=optimizer, grad_clip=grad_clip)


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    return _run_epoch(model, data_loader, criterion, device)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_history(path, history):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def fit(model, data, config, device, output_dir=None):
    """Notebook 和命令行共用。按验证 Loss 选最佳模型，最后只评估一次测试集。"""
    run_dir = resolve_output_dir(output_dir) / model.model_type
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    best_loss, best_epoch = float("inf"), 0
    best_valid = None
    started = time.perf_counter()

    _write_json(run_dir / "config.json", {
        "model_config": model.model_config,
        "experiment_config": asdict(config),
        "data": data.metadata,
        "device": str(device),
    })
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, data.loaders["train"], criterion, optimizer, device, config.grad_clip)
        valid_metrics = evaluate(model, data.loaders["valid"], criterion, device)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"valid_{key}": value for key, value in valid_metrics.items()})
        history.append(row)
        _write_history(run_dir / "history.csv", history)
        if valid_metrics["loss"] < best_loss:
            best_loss, best_epoch, best_valid = valid_metrics["loss"], epoch, valid_metrics
            torch.save({
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_config": model.model_config,
                "experiment_config": asdict(config),
                "vocab": data.vocab,
                "data": data.metadata,
                "best_epoch": best_epoch,
                "valid_metrics": best_valid,
            }, checkpoint_path)
        print(
            f"{model.model_type.upper()} | Epoch {epoch:02d}/{config.epochs:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_ppl={train_metrics['perplexity']:.2f} | "
            f"valid_loss={valid_metrics['loss']:.4f} valid_ppl={valid_metrics['perplexity']:.2f} | "
            f"valid_accuracy={valid_metrics['accuracy']:.2%}", flush=True,
        )

    train_seconds = time.perf_counter() - started
    best_model, _, _ = load_model(checkpoint_path, device)
    model.load_state_dict(best_model.state_dict())
    model.eval()
    test_metrics = evaluate(model, data.loaders["test"], criterion, device)
    metrics = {
        "model_type": model.model_type,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": train_seconds,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "experiment_config": asdict(config),
        "data": data.metadata,
        "valid": best_valid,
        "test": test_metrics,
    }
    _write_json(run_dir / "metrics.json", metrics)
    sample = generate_text(model, data.vocab, "the time traveller", num_words=50, seed=config.seed)
    (run_dir / "sample.txt").write_text(sample + "\n", encoding="utf-8")
    print(f"最佳 epoch: {best_epoch} | test_ppl={test_metrics['perplexity']:.2f} | 已保存: {run_dir}")
    return {"history": history, "metrics": metrics, "checkpoint_path": checkpoint_path, "run_dir": run_dir}
