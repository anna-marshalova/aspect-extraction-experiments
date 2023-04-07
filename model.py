import json
from importlib import import_module

from tf.keras import Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, Bidirectional, LSTM, Input
from tensorflow_addons.layers import CRF

from utils import MAX_LENGTH_FOR_TOKENIZER as MAX_LENGTH, num_labels


def get_model(model_name:str, num_labels:int=num_labels):
    with open('models.json', 'r', encoding='UTF-8') as js:
        models = json.load(js)
        model_config = models[model_name]
    model_class = getattr(import_module('transformers'), model_config['model_class'])
    config_class = getattr(import_module('transformers'), model_config['model_config'])

    if model_config['model_class'].endswith("ForTokenClassification"):
        config = config_class.from_pretrained(model_config["pretrained_model_name"], num_labels=num_labels)
        model = model_class.from_pretrained(model_config["pretrained_model_name"], config=config, **model_config['config'])
        model.layers[-1].activation = sigmoid

    elif model_config['model_class'].endswith("Model"):
        transformer_model = model_class.from_pretrained(model_config["pretrained_model_name"], model_config["config"])
        input_ids_in = Input(shape=(MAX_LENGTH,), name='input_token', dtype='int32')
        input_masks_in = Input(shape=(MAX_LENGTH,), name='masked_token', dtype='int32')
        if "frozen" in model_name:
            for layer in transformer_model.layers:
                layer.trainable = False
                for w in layer.weights: w._trainable = False

        X = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
        X = Dropout(model_config["dropout_rate"])(X)

        if "BiLSTM" in model_name:
            X = Bidirectional(LSTM(model_config["LSTM_size"], return_sequences=True, dropout=model_config["dropout_rate"], recurrent_dropout=model_config["dropout_rate"]))(X)

        if 'CRF' in model_name:
            X = CRF(num_labels)(X)[1]
        else:
            X = TimeDistributed(Dense(num_labels, activation='sigmoid'))(X)

        model = Model(inputs=[input_ids_in, input_masks_in], outputs=X)

    print(model.summary())
    return model





