import os
import numpy as np
import pandas as pd
from itertools import chain
from collections import OrderedDict
from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

from utils import ASPECTS_LIST, paths


class Evaluator:
    """Класс для оценки модели"""
    def __init__(self, predicted_labels:List[List[str]], true_labels:List[List[str]]):
        """
        :param predicted_labels: Список предсканных тэгов
        :param true_labels: Список истинных тэгов
        """
        self._class2tag = dict(enumerate(ASPECTS_LIST + ['O']))
        self._tag2class = dict(zip(self._class2tag.values(), self._class2tag.keys()))
        self.predicted_labels = self._vectorize_labels(predicted_labels)
        self.true_labels = self._vectorize_labels(true_labels)

    def _vectorize_label(self, label:str) -> np.array:
        """
        One-hot векторизация тэга
        :param label: Тэг в формате строки
        :return: Тэг в формате one-hot вектора
        """
        vector = np.zeros(len(self._tag2class))
        classes = [self._tag2class[tag] for tag in label.split('|')]
        vector[classes] = 1
        return vector

    def _vectorize_labels(self, labels:List[str]) -> np.array:
        """
        One-hot векторизация тэгов
        :param labels: Список тэгов в строковом формате
        :return: Массив one-hot векторов
        """
        return np.vstack([self._vectorize_label(label) for label in list(chain.from_iterable(labels))])

    def _unvectorize_labels(self, labels:np.array) -> List[str]:
        """
        Преобразование массива one-hot векторов в список строк
        :param labels: массив one-hot векторов
        :return: список тэгов в строковом формате
        """
        return ['|'.join([self._class2tag[cls] for cls in np.argwhere(label > 0).flatten()]) for label in labels]

    def confusion_matrix(self) -> np.array:
        """
        Построение confusion matrix
        :return: Confusion matrix (tp, fp, fn, tn)
        """
        confusion = self.true_labels + (self.predicted_labels * 2)
        confusion_matrix = np.vstack((
            # true positive: true = 1 & pred = 1 => true+2*pred = 1+2*1 = 3
            np.count_nonzero(confusion == 3, axis=0),
            # false positive: true = 0 & pred = 1 => true+2*pred = 0+2*1 = 2
            np.count_nonzero(confusion == 2, axis=0),
            # false negative: true = 1 & pred = 0 => true+2*pred = 1+2*0 = 1
            np.count_nonzero(confusion == 1, axis=0),
            # true negative: true = 0 & pred = 0 => true+2*pred = 0+2*0 = 0
            np.count_nonzero(confusion == 0, axis=0)
        ))
        return confusion_matrix

    def count_metrics_for_tag(self, tag_idx:int)-> List[float]:
        """
        Вычисление метрик для отдельного значения тэга
        :param tag: Индекс тэга в массиве
        :return: Метрики precision, recall, f1 и accuracy для заданного тэга
        """
        true = self.true_labels[:, tag_idx]
        pred = self.predicted_labels[:, tag_idx]
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)
        accuracy = balanced_accuracy_score(true, pred)
        return [precision, recall, f1, accuracy]

    def count_avg_metrics(self, average:str)-> List[float]:
        """
        Вычисление средних метрик для всех тэгов
        :param average: Параметр average для функций подсчета метрик из sklearn {‘micro’, ‘macro’, ‘weighted’}
        :return: Метрики precision, recall, f1 и accuracy, усредненные по всем тэгам
        """
        true = self.true_labels
        pred = self.predicted_labels
        precision = precision_score(true, pred, average=average)
        recall = recall_score(true, pred, average=average)
        f1 = f1_score(true, pred, average=average)
        accuracy = accuracy_score(true, pred)
        return [precision, recall, f1, accuracy]

    def count_metrics(self) -> np.array:
        """
        Построение массива с метриками
        :return: Массив с метриками precision, recall, f1 и accuracy для каждого из тэгов + микро- и макроусреднения по всем тэгам
        """
        metrics = []
        for tag in self._class2tag.keys():
            metrics.append(self.count_metrics_for_tag(tag))
        metrics.append(self.count_avg_metrics(average='micro'))
        metrics.append(self.count_avg_metrics(average='macro'))
        return np.vstack(metrics)

    def evaluate(self, **kwargs) -> pd.DataFrame:
        """
        Построение датафрейма с метриками
        :return: Датафрейм с метриками precision, recall, f1 и accuracy для каждого из тэгов + микро- и макроусреднения по всем тэгам
        """
        metrics = self.count_metrics()
        columns = ['Precision', 'Recall', 'F1', 'Accuracy']
        index = list(self._tag2class.keys()) + ['Micro', 'Macro']
        metrics_df = pd.DataFrame(metrics, columns=columns, index=index)
        metrics_df.columns.values[0] = 'Aspect'
        return metrics_df

    def build_confusion_matrix(self) -> pd.DataFrame:
        """
        Построение confusion matrix в формате датафрейма
        :return: Confusion matrix в формате датафрейма
        """
        confusion_matrix = self.confusion_matrix().T
        columns = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=columns, index=self._tag2class.keys())
        return confusion_matrix_df

    def save_metrcis(self, experiment_name:str, results_dir=paths['results'], **kwargs):
        """
        Сохранение метрик в файл
        :param experiment_name: Название эксперимента
        :param results_dir: Папка для сохранения
        :return: Датафрейм с метриками precision, recall, f1 и accuracy для каждого из тэгов + микро- и макроусреднения по всем тэгам
        """
        metrics_df = self.evaluate()
        path = os.path.join(results_dir, f'{experiment_name}.csv')
        metrics_df.to_csv(path)
        print(f'Metrics saved to {path}')
        return metrics_df

    def _make_label_chains(self, aspect_mask:np.array) -> List[Tuple[int, int]]:
        """
        Составление цепочек, относящихся к заданному аспекту. Вспомогательная функция для вычисления точности полного совпадения аспектов.
        :param aspect_mask: Маска аспекта: массив, где i-ое значение равно 1, если i-ый токен относится к данному аспекту.
        :return: Список извлеченных цепочек аспектов.Цепочка имеет вид [индекс начала аспекта, индекс конца аспекта]
        """
        # из маски получаем индексы токенов, относящихся к данному аспекту
        sparse_aspect_mask = np.arange(len(aspect_mask))[aspect_mask == 1]
        chains = []
        start, end = sparse_aspect_mask[0], sparse_aspect_mask[0]
        for value in sparse_aspect_mask:
            # если между индексами нет разрыва (например, 0 1 2 3), значит аспект продолжается
            if value - end <= 1:
                end = value
            # если между индексами есть разрыв (например, 0 1 2 3 10), значит аспект закончился. записываем границы в список
            else:
                chains.append(tuple([start, end]))
                start = value
                end = value
        if start and end:
            chains.append(tuple([start, end]))
        return chains

    def exact_match_accuracy(self)-> OrderedDict:
        """
        Вычисление точности полного совпадения аспектов: число полностью правильно извлеченных аспектов/число аспектов в тестовых данных
        Если хотя бы одно слово не вошло в аспект, или, наоборот, вошло лишнее, аспект считается извлеченным неправильно
        :return: Датафрейм с точностями полного совпадения для каждого аспекта + микро- и макроусреднения по всем тэгам
        """
        total_intersection, total_true_chains = 0, 0
        ema_dict = OrderedDict()
        for tag_idx, tag in self._class2tag.items():
                true_chains = set(self._make_label_chains(self.true_labels[:, tag_idx]))
                pred_chains = set(self._make_label_chains(self.predicted_labels[:, tag_idx]))
                # число правильно извлеченных аспектов
                intersection = len(true_chains&pred_chains)
                # число аспектов в тестовых данных
                num_true_chains = len(true_chains)
                total_intersection += intersection
                total_true_chains += num_true_chains
                ema_for_tag = intersection / num_true_chains if num_true_chains else 0.0
                ema_dict.update({tag: ema_for_tag})
        micro_ema = total_intersection/total_true_chains if total_true_chains else 0.0
        ema_dict.update({'Micro': micro_ema})
        ema_df = pd.DataFrame.from_dict(ema_dict, orient='index', columns=['Exact match accuracy'])
        ema_df.loc['Macro'] = ema_df[:-1].mean()
        return ema_df
