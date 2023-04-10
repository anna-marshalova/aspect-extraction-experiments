import json
from importlib import import_module

from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, Bidirectional, LSTM, Input
from tensorflow_addons.layers import CRF

from utils import MAX_LENGTH_FOR_TOKENIZER as MAX_LENGTH, paths, num_labels
from config import get_model_config


def get_model(model_name:str, num_labels:int=num_labels):
    """
    Создание модели
    :param model_name: Название модели
    :param num_labels: Число классов
    :return: Модель
    """
    model_config = get_model_config(model_name)

    if model_config.transformer_model.__name__.endswith("ForTokenClassification"):
        config = model_config.transformer_config.from_pretrained(model_config.pretrained_model_name, num_labels=num_labels)
        model = model_config.transformer_model.from_pretrained(model_config.pretrained_model_name, config=config, from_pt = model_config.from_pt)
        model.layers[-1].activation = sigmoid

    elif model_config.transformer_model.__name__.endswith("Model"):
        transformer_model = model_config.transformer_model.from_pretrained(model_config.pretrained_model_name, from_pt = model_config.from_pt)
        input_ids_in = Input(shape=(MAX_LENGTH,), name='input_token', dtype='int32')
        input_masks_in = Input(shape=(MAX_LENGTH,), name='masked_token', dtype='int32')
        if "frozen" in model_name:
            for layer in transformer_model.layers:
                layer.trainable = False
                for w in layer.weights: w._trainable = False

        X = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
        X = Dropout(model_config.dropout_rate)(X)
        if model_config.classify_sequence:
            X = X[:, 0, :] #получаем cls-вектор

        if model_config.bilstm_size > 0:
            X = Bidirectional(LSTM(model_config.bilstm_size, return_sequences=True, dropout=model_config.dropout_rate, recurrent_dropout=model_config.dropout_rate))(X)

        if model_config.add_crf:
            X = CRF(num_labels)(X)[1]
        elif model_config.classify_sequence:
            X = Dense(num_labels, activation='sigmoid')(X)
        else:
            X = TimeDistributed(Dense(num_labels, activation='sigmoid'))(X)

        model = Model(inputs=[input_ids_in, input_masks_in], outputs=X)

    print(model.summary())
    return model





