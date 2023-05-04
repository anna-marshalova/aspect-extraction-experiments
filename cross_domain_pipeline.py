import logging
import pandas as pd
from typing import List, Tuple
from itertools import chain
from tqdm.autonotebook import tqdm
from tensorflow.keras.callbacks import History

from model import get_model
from trainer import Trainer
from predictor import Predictor
from evaluator import Evaluator
from utils import RANDOM_STATE, DOMAINS, paths


class CrossDomainPipeline:
    """ Класс для проведения кросс-доменных экспериментов"""

    def __init__(self, test_domain: str, samples: Tuple[List[List[List[str]]], List[List[List[str]]]], model_name: str,
                 experiment_series_name: str, weights_dir: str = paths['cross_domain_weights'],
                 domains: List[str] = DOMAINS, additional_samples: Tuple[List[List[str]], List[List[str]]] = ([],[]), random_state = RANDOM_STATE, model = None):
        """
        :param test_domain: Название предметной области, на которой будет тестироваться модель
        :param domains: Список всех предметных областей
        :param samples: Кортеж, в котором первый элемент списки текстов, второй - списки тэгов.
        Тексты (и, соответственно, тэги) представляют собой списки токенов в списках текстов в списках доменов).
        Чтобы получить такие списки, нужно в методе load_dataset() класса DataLoader указать mode='cross_domain')
        :param model_name: Название модели
        :param experiment_series_name: Название серии кросс-доменных экспериментов
        :param weights_dir: Путь к папке с весами модели
        :param additional_samples: Дополнительные данные для обучения
        :param random_state: Случайное состояние для деления данных
        :param model: Модель, созданная заранее
        """
        self._weights_dir = weights_dir
        self._domains = domains
        self._test_domain = test_domain
        self._model_name = model_name
        if model:
            self._model = model
        else:
            self._model = get_model(self._model_name)
        self._experiment_series_name = experiment_series_name
        self.train_samples, self.test_samples, self.train_labels, self.test_labels = self._split_data(samples)
        self.additional_samples, self.additional_labels = additional_samples
        self.random_state = random_state
        self._experiment_name = f'{self._test_domain}_{self._experiment_series_name}_{self._model_name}'
        if self.random_state!=RANDOM_STATE:
            self._experiment_name += f'_{self.random_state}'

    def _split_data(self, samples: Tuple[List[List[List[str]]], List[List[List[str]]]]) -> Tuple[
        List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:
        """
        Разделение данных по фолдам (предметным областям)
        :param samples: Кортеж, в котором первый элемент списки текстов, второй - списки тэгов.
        :param labels:
        :return: Данные, разделенные на обучающие и тестовые
        """
        text_samples, labels = samples
        # выбираем тестовый фолд из выборки
        num_test_domain = self._domains.index(self._test_domain)
        test_samples = text_samples[num_test_domain]
        test_labels = labels[num_test_domain]
        # все остальные фолды объединяем в трейн
        train_samples = list(chain.from_iterable(text_samples[:num_test_domain])) + list(
            chain.from_iterable(text_samples[num_test_domain + 1:]))
        train_labels = list(chain.from_iterable(labels[:num_test_domain])) + list(
            chain.from_iterable(labels[num_test_domain + 1:]))
        assert len(train_samples) == len(train_labels)
        assert len(test_samples) == len(test_labels)
        print(f'Domain: {self._test_domain}. Train samples: {len(train_samples)}. Test samples: {len(test_samples)}.')
        return train_samples, test_samples, train_labels, test_labels

    def train(self, save_weights: bool = True, **kwargs) -> History:
        """
        Обучение модели
        :param save_weights: Сохранять ли веса на диск
        :return: История обучения для анализа и построения графиков
        """
        logging.basicConfig(level=logging.ERROR)
        trainer = Trainer(
            samples=(self.train_samples+self.additional_samples, self.train_labels+self.additional_labels),
            experiment_name=self._experiment_name,
            model_name=self._model_name,
            model=self._model,
            random_state=self.random_state)
        history = trainer.train(save_weights=save_weights)
        return history

    def __set_predictor(self):
        weights_filename = f'{self._experiment_name}_weights.h5'
        self.predictor = Predictor(model_name=self._model_name, weights_filename=weights_filename, model=self._model)

    def __set_evaluator(self, predicted_labels: List[List[str]]):
        """
        :param predicted_labels: Список предсказанных тэгов (отдельно для каждого текста)
        """
        self.evaluator = Evaluator(predicted_labels, self.test_labels)

    def evaluate(self, use_heuristics: bool = True, save_metrics: bool = True, **kwargs) -> pd.DataFrame:
        """
        Оценка модели
        :param use_heuristics: Применять ли к полученному результату эвристики
        :param save_metrics: Сохранять ли метрики на диск
        :return: Датафрейм с метриками precision, recall, f1 и accuracy для каждого из тэгов + микро- и макроусреднения по всем тэгам
        """
        self.__set_predictor()
        print('Making predictions')
        predictions = [self.predictor.extract(text, use_heuristics=use_heuristics) for text in tqdm(self.test_samples)]
        predicted_labels = [[label for token, label in text] for text in predictions]
        self.__set_evaluator(predicted_labels)
        if save_metrics:
            return self.evaluator.save_metrcis(experiment_name=f'{self._experiment_name}',
                                               results_dir=paths['cross_domain_results'])
        return self.evaluator.evaluate()

    def pipeline(self, **kwargs):
        """
        Пайплан для обучения и оценки модели
        :param save_weights: Сохранять ли веса на диск
        :param use_heuristics: Применять ли к полученному результату эвристики
        :param save_metrics: Сохранять ли метрики на диск
        :return: ``history``: История обучения для анализа и построения графиков
                 ``metrics_df``: Датафрейм с метриками precision, recall, f1 и accuracy для каждого из тэгов + микро- и макроусреднения по всем тэгам
        """
        history = self.train(**kwargs)
        metrics_df = self.evaluate(**kwargs)
        return history, metrics_df
