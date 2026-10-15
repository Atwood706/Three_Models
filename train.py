"""python train.py --model all：公平训练三个单词级语言模型。"""
import argparse

from config import ExperimentConfig, MODEL_TYPES, get_device, resolve_data_path, resolve_output_dir
from src.dataset import prepare_data
from src.engine import fit, seed_everything
from src.model_factory import create_model
from src.plotting import comparison_table, plot_history


def main():
    parser = argparse.ArgumentParser(description="Time Machine 单词级 next-token prediction")
    parser.add_argument("--model", choices=(*MODEL_TYPES, "all"), default="all")
    parser.add_argument("--data-path", help="pg35.txt 路径；也可使用 TIME_MACHINE_DATA_PATH")
    parser.add_argument("--output-dir", help="输出根目录；下面按 rnn / lstm / gru 分目录")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    defaults = ExperimentConfig()
    for name in ("seed", "min_freq", "seq_len", "batch_size", "embed_size", "hidden_size", "num_layers", "epochs"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("train_ratio", "valid_ratio", "dropout", "learning_rate", "grad_clip"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    args = parser.parse_args()
    config = ExperimentConfig(**{name: getattr(args, name) for name in vars(defaults)})
    data_path = resolve_data_path(args.data_path)
    output_dir = resolve_output_dir(args.output_dir)
    device = get_device(args.device)
    print(f"文本: {data_path}\n输出: {output_dir}\n设备: {device}")
    selected = MODEL_TYPES if args.model == "all" else (args.model,)
    for model_type in selected:
        # 每个模型重新设种子和 DataLoader 的生成器，保证数据顺序一致。
        seed_everything(config.seed)
        data = prepare_data(data_path, config)
        print(f"Tokens: {data.metadata['total_tokens']} | Vocab: {len(data.vocab)}")
        model = create_model(model_type, len(data.vocab), config)
        result = fit(model, data, config, device, output_dir)
        figure = plot_history(result["history"])
        figure.savefig(result["run_dir"] / "learning_curves.png", dpi=150)
        from matplotlib import pyplot as plt
        plt.close(figure)
    if args.model == "all":
        table = comparison_table(output_dir)
        table.to_csv(output_dir / "comparison.csv", index=False)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
