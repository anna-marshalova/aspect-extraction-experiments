import json
from dataclasses import dataclass
from importlib import import_module
import tensorflow as tf
import transformers
from utils import paths


@dataclass
class ModelConfig:
    """Гипепараметры и конфигурации модели"""
    pretrained_model_name: str = "bert-base-multilingual-cased"
    transformer_config: type = transformers.BertConfig
    transformer_model: type = transformers.TFBertModel
    transformer_tokenizer: type = transformers.BertTokenizer
    from_pt: bool = False
    dropout_rate: float = 0.1
    bilstm_size: int = 0
    learning_rate: float = 1e-4
    loss: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy
    freeze_weights: bool = False
    classify_sequence: bool = False
    add_crf: bool = False


def get_model_config(model_name: str):
    # TODO
    with open(paths['model_config'], 'r', encoding='UTF-8') as js:
        model_config_dict = json.load(js)[model_name]
        for key, value in model_config_dict.items():
            if key in ["transformer_config", "transformer_model", "transformer_tokenizer", "loss"]:
                model_config_dict[key] = get_class(value)
        return ModelConfig(**model_config_dict)


def get_class(full_class_name: str):
    #TODO
    full_class_name = full_class_name.split('.')
    module_name = '.'.join(full_class_name[:-1])
    return getattr(import_module(module_name), full_class_name[-1])
