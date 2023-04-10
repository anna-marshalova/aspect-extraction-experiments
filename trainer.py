import os
import json
import numpy as np
from importlib import import_module
from typing import List, Tuple, Generator
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, History
from tensorflow.keras.metrics import CategoricalAccuracy

from utils import MAX_LENGTH_FOR_TOKENIZER as MAX_LENGTH, RANDOM_STATE, paths, tag2class, sort_dataset
from model import get_model
from vectorizer import Vectorizer
from config import get_model_config

class Trainer:
    """Класс для обучения модели"""
    def __init__(self, samples:Tuple[List[List[str]],List[List[str]]], model_name:str, experiment_name:str, weights_dir:str=paths['weights'], load_weights:bool=False, model=None, vectorizer:Vectorizer=None, batch_size: int = 16):
        """
        :param samples: Кортеж, в котором первый элемент список текстов, второй - список тэгов
        :param model_name: Название модели
        :param experiment_name: Название эксперимента
        :param weights_dir: Путь к папке для загрузки и сохранения весов модели
        :param load_weights: Загружать ли веса с диска
        :param model: Модель, созданная заранее
        :param vectorizer: Векторизатор, созданный заранее
        :param batch_size: Размер батча
        """
        self._tag2class = tag2class
        if model:
            self._model = model
        else:
            self._model = get_model(model_name)
        self._model_name = model_name
        if vectorizer:
            self._vectorizer = vectorizer
        else:
            self._vectorizer = Vectorizer(self._model_name)
        
        self._weights_dir = weights_dir
        if not os.path.exists(self._weights_dir):
            os.makedirs(self._weights_dir, exist_ok=True)
        weights_filename = f'{experiment_name}_weights.h5'
        self._path_to_weights = os.path.join(self._weights_dir, weights_filename)
        if load_weights:
            self._model.load_weights(self._path_to_weights)

        self._batch_size = batch_size
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(samples[0], samples[1],
                                                                                  random_state=RANDOM_STATE)
        self._X_train, self._y_train = sort_dataset(self._X_train, self._y_train)
        self._X_val, self._y_val = sort_dataset(self._X_val, self._y_val)
        print(f'{len(self._X_train)} train samples, {len(self._X_val)} val samples')

        self._steps_per_epoch = int(len(self._X_train) / self._batch_size)
        self._validation_steps = int(len(self._X_val) / self._batch_size)

        self._num_of_train_samples = self._steps_per_epoch * self._batch_size
        self._num_of_val_samples = self._validation_steps * self._batch_size

        self._set_config()

    def get_model(self):
        return self._model
    def _set_config(self):
        self._model_config = get_model_config(self._model_name)
        self._learning_rate = self._model_config.learning_rate
        self._schedule = ExponentialDecay(initial_learning_rate=self._learning_rate,
                                         decay_steps=2 * self._steps_per_epoch, decay_rate=0.9)
        self._optimizer = Adam(learning_rate=self._schedule)
        self._loss = self._model_config.loss()

    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    def _generate_samples(self, samples:List[str], labels:List[str], num_of_samples:int)->Generator[np.array, np.array, np.array]:
        """
        Генерация батча
        :param samples: Список текстов для батча
        :param labels: Список тэгов для батча
        :param num_of_samples: Количество экземпляров в батче
        :return: Батч: генератор, содержащий векторизованные тексты, маски к ним и тэги
        """
        i = 0
        while True:
            texts = samples[i:i + self._batch_size]
            y_labels = labels[i:i + self._batch_size]
            X_ids = []
            X_masks = []
            y = []
            i += self._batch_size
            # если архитектура модели позволяет, выбираем свою максимальную длинну для каждого батча
            if self._model_config.transformer_model.__name__.endswith("ForTokenClassification"):
                max_length = max([len(text) for text in texts]) * 2
            else:
                max_length = MAX_LENGTH
            for text, token_labels in zip(texts, y_labels):
                _, input_ids, input_masks, tags = self._vectorizer.vectorize(text, token_labels, max_length=max_length)
                X_ids.append(np.array(input_ids))
                y.append(tags)
                X_masks.append(np.array(input_masks))
            yield [np.asarray(X_ids, dtype='int32'), np.asarray(X_masks, dtype='int32')], np.array(y)
            if i == num_of_samples:
                i = 0

    def train(self, save_weights:bool=True, num_epochs:int = 300, patience:int = 5) -> History:
        """
        Обучение модели
        :param save_weights: Сохранять ли веса на диск
        :param num_epochs: Количество эпох обучения
        :param patience: Количество эпох, в течение которых обучение продолжается при отсутствии уменьшения функции потерь на валидации
        :return: История обучения для анализа и построения графиков
        """
        saver = ModelCheckpoint(self._path_to_weights, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                save_weights_only=True)
        stopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto',
                                restore_best_weights=True)
        scheduler = LearningRateScheduler(self._schedule)
        callbacks = [stopper, scheduler]
        if save_weights:
            callbacks.append(saver)
        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=[CategoricalAccuracy()]
        )
        history = self._model.fit(
            self._generate_samples(self._X_train, self._y_train, self._num_of_train_samples),
            epochs=num_epochs,
            validation_data=self._generate_samples(self._X_val, self._y_val, self._num_of_val_samples),
            steps_per_epoch=self._steps_per_epoch,
            validation_steps=self._validation_steps,
            verbose=1,
            callbacks=callbacks,
        )
        return history
