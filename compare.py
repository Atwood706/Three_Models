"""读取已经完成的训练结果，不重新训练。"""
import argparse

from config import MODEL_TYPES, resolve_output_dir
from src.plotting import comparison_table


def main():
    parser = argparse.ArgumentParser(description="比较 RNN / LSTM / GRU")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    table = comparison_table(output_dir)
    if table.empty:
        raise SystemExit("没有找到 metrics.json，请先运行 train.py。")
    missing = set(MODEL_TYPES) - set(table["model"].str.lower())
    if missing:
        print(f"尚未训练: {', '.join(sorted(missing))}；下面显示已完成的模型。")
    table.to_csv(output_dir / "comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"比较结果已保存到: {output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
