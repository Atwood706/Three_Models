"""为命令行与 checkpoint 加载选择模型；模型本身在三个独立文件中。"""
from src.model_RNN import TimeMachineRNN
from src.model_LSTM import TimeMachineLSTM
from src.model_GRU import TimeMachineGRU


MODEL_CLASSES = {
    "rnn": TimeMachineRNN,
    "lstm": TimeMachineLSTM,
    "gru": TimeMachineGRU,
}


def create_model_from_config(model_config):
    parameters = dict(model_config)
    model_type = parameters.pop("model_type")
    if model_type not in MODEL_CLASSES:
        raise ValueError("model_type 必须为 rnn、lstm 或 gru。")
    return MODEL_CLASSES[model_type](**parameters)


def create_model(model_type, vocab_size, config):
    return create_model_from_config({
        "model_type": model_type,
        "vocab_size": vocab_size,
        "embed_size": config.embed_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    })
