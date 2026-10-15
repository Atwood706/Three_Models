# Time Machine：单词级下一个 token 预测

沿用 `NLP_Disaster_Tweets` 的学习结构，用 H. G. Wells 的《时间机器》训练 RNN、LSTM、GRU 三个语言模型。代码基于 PyTorch 内置循环层，公共逻辑放在 `src/`，Notebook 用中文逐步解释。原始 `pg35.txt` 保留在项目根目录。

## 为什么可以按单词分？

字符和单词都可以作为 token。本项目选择**单词级**：输入 `the time`，预测下一个单词，例如 `traveller`。相比字符级，词表更大、稀有词更多，小语料上更容易过拟合，但没有技术障碍。训练词表中出现次数小于 `min_freq` 的词以及验证、测试集中的新词会映射成 `<unk>`。

当前 tokenizer 统一小写，保留词内撇号（`don't`、`traveller's`），去掉数字和标点，连字符会把词分开。输出是空格连接的小写英文单词，不恢复大小写和标点，也不使用预训练 tokenizer、预训练模型或外部语料。

## 目录与阅读顺序

```text
Time_Machine/
├── pg35.txt                    # 已有的原始文本
├── config.py                   # 实验参数、数据路径、输出路径、设备选择
├── train.py                    # 一个模型或三个模型的训练入口
├── generate.py                 # 下一词候选与连续生成
├── compare.py                  # 汇总已经保存的实验结果
├── requirements.txt
├── src/
│   ├── preprocessing.py        # 正文提取、分词、词表、数值化
│   ├── dataset.py              # 顺序切分、X/y 窗口、DataLoader
│   ├── model_RNN.py            # 独立的 RNN 模型定义
│   ├── model_LSTM.py           # 独立的 LSTM 模型定义
│   ├── model_GRU.py            # 独立的 GRU 模型定义
│   ├── model_factory.py        # 命令行和模型加载使用的模型选择入口
│   ├── engine.py               # 训练、验证、最佳模型保存、测试
│   ├── generation.py           # 模型加载、自回归生成
│   └── plotting.py             # 学习曲线与模型比较
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_dataset.ipynb
│   ├── 04_rnn.ipynb
│   ├── 05_lstm.ipynb
│   └── 06_gru.ipynb
├── tests/                      # 数据边界、因果性、训练与保存/加载检查
└── outputs/                    # 运行训练后生成
    ├── rnn/
    ├── lstm/
    └── gru/
```

前三个 Notebook 为共享准备步骤，后三个各自包含模型结构、张量形状、损失计算、训练、评估、加载和文本生成。每个 Notebook 都可以在新 kernel 中独立从头运行，不需要继承其他 Notebook 的变量。所有 Notebook 使用相同的默认配置，分别运行后可比较结果。

三个模型各自直接继承 `nn.Module`，在对应的 `model_RNN.py`、`model_LSTM.py`、`model_GRU.py` 中定义构造函数、循环层和前向传播，可以分别修改。模型 Notebook 直接导入各自的类；`model_factory.py` 只负责命令行训练和 checkpoint 加载时选择类。默认超参数仍由 `ExperimentConfig` 提供。

## 与二元分类项目的区别

| 内容 | Disaster Tweets | Time Machine |
| --- | --- | --- |
| 样本 | 一条推文 | 连续的单词窗口 |
| 标签 | 一个 0/1 标签 | 输入向后移一个位置的单词 ID |
| 循环层输出使用方式 | 最后隐藏状态 | 每个时间步的输出 |
| logits 形状 | `[batch]` | `[batch, seq_len, vocab_size]` |
| 损失 | `BCEWithLogitsLoss` | `CrossEntropyLoss` |
| 评估 | 分类准确率 | 单词交叉熵、困惑度、下一词准确率 |
| 推断 | 判断是不是灾难 | 下一词候选及连续续写 |

例如：

```text
原序列：the time traveller was speaking
X：     the time traveller was
y：     time traveller was speaking
```

每个时间步只能看到当前及之前的输入。三个循环层都为**单向**，避免用未来单词预测当前目标。

## 数据与训练约定

- 去掉 Gutenberg 前后说明、许可证，以及本书的标题和目录，保留正文顺序。
- 先按正文 token 顺序切成 80% 训练、10% 验证、10% 测试，再构造窗口；不同数据集之间没有共享原始 token 位置。
- 词表只用训练集建立；保留 `<pad>=0`、`<unk>=1`，但当前固定长度窗口不需要 padding。
- `seq_len=32`；窗口步长为 32。相邻窗口衔接处共享一个原始 token，但作为目标的 token 位置不重复。每个 split 最后不足一个完整窗口的目标会被丢弃，其数量写入数据统计。
- 训练 DataLoader 可以打乱完整窗口，窗口内部的 token 顺序保持不变。每个 batch 重新初始化隐藏状态；因此这是固定窗口训练，不跨 batch 传递状态。
- 模型输入 `[B,T]` 经 Embedding 成 `[B,T,E]`，循环层输出 `[B,T,H]`，Linear 输出 `[B,T,V]`。计算交叉熵前展开为 `[B*T,V]` 和 `[B*T]`。
- 三个模型的随机种子、词表、数据顺序和训练超参数一致；相同隐藏维度下，它们的参数量不同，比较表会记录参数量。
- 默认 Embedding 100、隐藏维度 128、单层、Adam、学习率 0.001、梯度裁剪 1.0、20 epochs；均可修改。固定种子不保证不同设备间结果完全相同。
- 按验证集 Loss 选择最佳 epoch，训练结束后加载该模型，再评估一次测试集。调整超参数时看验证集，测试集只用于最终报告。

困惑度 `PPL = exp(平均 token 交叉熵)`，越低越好。平均损失按**目标 token 数量**加权。`accuracy` 包含 `<unk>` 目标；`known_accuracy` 仅统计真实词表内单词；同时报告 `unknown_target_rate`。这份语料较小，不能保证生成长文流畅，也不能先验认定 LSTM 或 GRU 一定优于 RNN。

## Mac 本地

在项目目录使用已有的 PyTorch 环境（例如原项目的 `DL` 环境）：

```bash
cd /Users/macbook/Projects/d2l_extra_projects/Time_Machine
conda activate DL
# 仅在缺少依赖时安装
python -m pip install -r requirements.txt
jupyter lab notebooks
```

在 Jupyter 中选择安装了上述依赖的 kernel。默认自动按 CUDA → MPS → CPU 的顺序选择可用设备；Mac 通常使用 MPS，没有可用 MPS 时使用 CPU。Notebook 会显示实际设备，也可把 `get_device("auto")` 改成 `get_device("cpu")`。

命令行训练与预测：

```bash
python train.py --model all
python train.py --model rnn --epochs 20
python generate.py --checkpoint outputs/rnn/best.pt --prefix "the time traveller" --num-words 50
python compare.py
```

先检查完整流程时，可以运行小配置；使用独立输出目录，避免覆盖正式结果：

```bash
python train.py --model all --epochs 1 --embed-size 16 --hidden-size 24 --batch-size 64 --device cpu --output-dir outputs/smoke
python compare.py --output-dir outputs/smoke
```

## Kaggle

项目代码与文本都需要可访问：只上传 `pg35.txt` 而不提供 `src/` 和 `config.py` 无法运行这些 Notebook。

1. 把整个 `Time_Machine` 目录作为 Kaggle Dataset 上传，包括 `src/`、入口脚本和 Notebook。也可以分别上传代码和文本。
2. 在 Kaggle Notebook 中添加该数据集，在需要 GPU 时启用 GPU。
3. 将项目复制到可写的 `/kaggle/working/Time_Machine`。挂载路径需要替换为你自己的实际路径。

```python
from pathlib import Path
import shutil

source = Path("/kaggle/input/YOUR_DATASET/Time_Machine")
destination = Path("/kaggle/working/Time_Machine")
shutil.copytree(source, destination, dirs_exist_ok=True)
```

随后在该目录运行 `train.py`，或将对应 Notebook 的单元格复制到 Kaggle Notebook。导入公共代码前设置项目路径：

```python
import os
os.environ["TIME_MACHINE_PROJECT_ROOT"] = "/kaggle/working/Time_Machine"
# 只有当文本单独存放或有多个 pg35.txt 时才需指定这一项
# os.environ["TIME_MACHINE_DATA_PATH"] = "/kaggle/input/YOUR_TEXT_DATASET/pg35.txt"
```

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "train.py", "--model", "all"],
    cwd="/kaggle/working/Time_Machine",
    check=True,
)
```

默认输出写到 `/kaggle/working/time_machine`，可在运行后下载。未找到项目根目录中的文本时，程序会搜索 `/kaggle/input/**/pg35.txt`；若有多个匹配，会要求明确指定。

## 服务器

把整个目录上传到服务器并安装依赖。若服务器有 NVIDIA GPU，使用与服务器驱动匹配的 PyTorch CUDA 构建，安装方式参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。CPU 服务器也可以训练。

```bash
cd /path/to/Time_Machine
python -m pip install -r requirements.txt
python train.py --model all --device auto --data-path /path/to/pg35.txt --output-dir /path/to/results
```

显式 `--device cuda` / `mps` 会检查设备可用性；不可用时会报错，便于发现环境配置问题。代码不使用硬编码的本机数据路径，也不依赖多进程 DataLoader。

可通过环境变量指定默认数据和输出位置，Notebook 与脚本共用：

```bash
export TIME_MACHINE_DATA_PATH=/path/to/pg35.txt
export TIME_MACHINE_OUTPUT_DIR=/path/to/results
```

## 输出与推断

每个模型目录包含：

- `best.pt`：最佳模型权重、模型配置、实验配置、词表、数据统计和最佳 epoch。
- `history.csv`：逐 epoch 的训练和验证指标。
- `metrics.json`：最佳验证指标、最终测试指标、参数量、设备、训练耗时。
- `config.json`：本次实验配置。
- `sample.txt`：用最佳模型生成的示例文本。
- `learning_curves.png`：学习曲线。

命令行 `--model all` 还会保存 `comparison.csv`。Notebook 训练后也会保存图，并展示目前已训练模型的比较表。比较程序检查语料、词表和实验配置是否一致，不会把不同设置的结果直接放到同一个表里；表格按验证困惑度排序。

同一输出目录内重跑同一模型会覆盖该模型的上次结果，想保留多次实验请用不同的 `--output-dir`。这些 checkpoint 用于推断，不包含优化器状态，不支持直接恢复中断的训练。

预测命令同时展示下一词候选和续写文本：

```bash
python generate.py --checkpoint outputs/lstm/best.pt --prefix "the machine" --num-words 40 --temperature 0.8 --top-k 20
# 贪心生成，每步选择最高分单词
python generate.py --checkpoint outputs/gru/best.pt --prefix "the time" --temperature 0
```

生成时先用 prefix 初始化隐藏状态，再把每次生成的单词反馈给模型，并保留隐藏状态。`temperature=0` 为贪心预测；正数用于温度采样；`top_k=0` 不限制候选数量。生成与候选展示会屏蔽 `<pad>` 和 `<unk>` 并重新归一化，显示的概率是这个条件下的概率；训练、验证和测试损失仍使用完整词表。prefix 中不在词表的单词会映射成 `<unk>`，命令行会提示。

## 检查

```bash
python -m unittest discover -s tests -v
```

测试检查正文清理、单词 token、训练词表边界、顺序划分、目标右移、三个模型的因果性，以及训练、最佳模型保存/加载和可复现生成。测试输出保存在临时目录，不覆盖 `outputs/`。

本地验证环境为 Python 3.8.20 / PyTorch 2.4.1 / CPU：六个 Notebook 的格式与语法均已检查，全部代码单元均在独立进程中运行通过（模型训练在验证副本中缩短为 1 epoch、Embedding 16、隐藏维度 24）。命令行三个模型也已使用完整正文完成相同配置的短训练，并检查了预测、生成和结果比较。CUDA / MPS 设备选择以及 Kaggle 挂载路径通过模拟检查；尚未在实际 GPU、MPS 或 Kaggle 环境中运行。上述短训练用于验证流程，不代表模型效果或正式排名。

接口参考：[RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)、[LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)、[GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)、[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
