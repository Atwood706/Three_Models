"""使用已保存的最佳模型预测下一个单词并生成文本。"""
import argparse

from config import get_device
from src.generation import generate_text, load_model, predict_next
from src.preprocessing import tokenize


def main():
    parser = argparse.ArgumentParser(description="预测下一个单词 / 文本续写")
    parser.add_argument("--checkpoint", required=True, help="训练输出的 best.pt")
    parser.add_argument("--prefix", default="the time traveller")
    parser.add_argument("--num-words", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 为贪心预测")
    parser.add_argument("--top-k", type=int, default=20, help="0 为全部普通单词")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    model, vocab, checkpoint = load_model(args.checkpoint, get_device(args.device))
    print(f"模型: {model.model_type.upper()} | 最佳 epoch: {checkpoint['best_epoch']}")
    unseen = sorted(set(tokenize(args.prefix)) - set(vocab))
    if unseen:
        print(f"输入中的未登录词会映射到 <unk>: {unseen}")
    print("下一个单词的候选（屏蔽特殊 token 后的概率）:")
    for candidate in predict_next(model, vocab, args.prefix, args.candidates):
        print(f"  {candidate['token']:<18} {candidate['probability']:.2%}")
    print("\n生成文本:")
    print(generate_text(model, vocab, args.prefix, args.num_words, args.temperature, args.top_k, args.seed))


if __name__ == "__main__":
    main()
