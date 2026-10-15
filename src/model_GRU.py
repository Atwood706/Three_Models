"""GRU 单词级语言模型：Embedding → GRU → 词表 logits。"""
from torch import nn


class TimeMachineGRU(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=128,
                 num_layers=1, dropout=0.0):
        super().__init__()
        self.model_type = "gru"
        self.model_config = {
            "model_type": self.model_type,
            "vocab_size": vocab_size,
            "embed_size": embed_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.recurrent = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, X, state=None):
        embedded = self.embedding(X)            # [batch, steps, embed_size]
        output, hidden = self.recurrent(embedded, state)
        logits = self.fc(self.dropout(output))  # [batch, steps, vocab_size]
        return logits, hidden
