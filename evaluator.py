import os
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

from utils import ASPECTS_LIST, tag2class, class2tag, paths


class Evaluator:
    def __init__(self, predicted_labels:List[str], true_labels:List[str]):
        """
        :param predicted_labels: Список предсканных тэгов
        :param true_labels: Список истинных тэгов
        """
        self.class2tag = dict(enumerate(ASPECTS_LIST + ['O']))
        self.tag2class = dict(zip(self.class2tag.values(), self.class2tag.keys()))
        self.flatten = lambda nested_list: [item for nest in nested_list for item in nest]
        self.predicted_labels = self.vectorize_labels(predicted_labels)
        self.true_labels = self.vectorize_labels(true_labels)
        self.num_pattern = re.compile('\d*')

    def vectorize_label(self, label:str) -> np.array:
        """
        One-hot векторизация тэга
        :param label: Тэг в формате строки
        :return: Тэг в формате one-hot вектора
        """
        vector = np.zeros(len(self.tag2class))
        classes = [self.tag2class[tag] for tag in label.split('|')]
        vector[classes] = 1
        return vector

    def vectorize_labels(self, labels:List[str]) -> np.array:
        """
        One-hot векторизация тэгов
        :param labels: Список тэгов в строковом формате
        :return: Массив one-hot векторов
        """
        return np.vstack([self.vectorize_label(label) for label in self.flatten(labels)])

    def unvectorize_labels(self, labels:np.array) -> List[str]:
        """
        Преобразование массива one-hot векторов в список строк
        :param labels: массив one-hot векторов
        :return: список тэгов в строковом формате
        """
        return ['|'.join([self.class2tag[cls] for cls in np.argwhere(label > 0).flatten()]) for label in labels]

    def confusion_matrix(self) -> np.array:
        """
        Построение confusion matrix
        :return: Confusion matrix (tp, fp, fn, tn)
        """
        confusion = self.true_labels + (self.predicted_labels * 2)
        confusion_matrix = np.vstack((
            np.count_nonzero(confusion == 3, axis=0),
            np.count_nonzero(confusion == 2, axis=0),
            np.count_nonzero(confusion == 1, axis=0),
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
        :return: Массив с метриками precision, recall, f1 и accuracy для каждого из тэгов + макро- и микроусреднения по всем тэгам
        """
        metrics = []
        for tag in self.class2tag.keys():
            metrics.append(self.count_metrics_for_tag(tag))
        metrics.append(self.count_avg_metrics(average='macro'))
        metrics.append(self.count_avg_metrics(average='micro'))
        return np.vstack(metrics)

    def evaluate(self) -> pd.DataFrame:
        """
        Построение датафрейма с метриками
        :return: Датафрейм с метриками precision, recall, f1 и accuracy для каждого из тэгов + макро- и микроусреднения по всем тэгам
        """
        metrics = self.count_metrics()
        columns = ['Precision', 'Recall', 'F1', 'Accuracy']
        index = list(self.tag2class.keys()) + ['Macro', 'Micro']
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
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=columns, index=self.tag2class.keys())
        return confusion_matrix_df

    def save_metrcis(self, experiment_name:str, dir_path=paths['results']):
        """
        Сохранение метрик в файл
        :param experiment_name: Название эксперимента
        :param dir_path: Папка для сохранения
        """
        metrics_df = self.evaluate()
        path = os.path.join(dir_path, f'{experiment_name}.csv')
        metrics_df.to_csv(path)
        print(f'Metrics saved to {path}')
        return metrics_df

    def make_label_chains(self, aspect_mask:List[str]) -> List[Tuple[str, List[str]]]:
        """
        Составление цепочек аспектов. Вспомогательная функция для вычисления точности полного совпадения аспектов.
        :param labels: Список тэгов
        :return: Список извлеченных цепочек аспектов. Пример цепочки: ('Method', ['Method_0', 'Method_1', 'Method_2'])
        """
        sparse_aspect_mask = np.arange(len(aspect_mask))[aspect_mask == 1]
        chains = []
        start, end = sparse_aspect_mask[0], sparse_aspect_mask[0]
        for value in sparse_aspect_mask:
            if value - end <= 1:
                end = value
            else:
                chains.append(tuple([start, end]))
                start = value
                end = value
        if start and end:
            chains.append(tuple([start, end]))
        return chains

    def exact_match_accuracy(self)-> OrderedDict:
        total_intersection, total_true_chains = 0, 0
        ema_dict = OrderedDict()
        for tag_idx, tag in self.class2tag.items():
                true_chains = set(self.make_label_chains(self.true_labels[:, tag_idx]))
                pred_chains = set(self.make_label_chains(self.predicted_labels[:, tag_idx]))
                intersection = len(true_chains&pred_chains)
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

import json
if __name__ == '__main__':
    with open('train_preds.json') as file:
        train_result= json.load(file)
    with open('train_labels.json') as file:
        train_labels = json.load(file)
    train_predicted_labels = [[label for token, label in text] for text in train_result]
    evaluator = Evaluator(train_predicted_labels, train_labels)
    #OrderedDict([('Task', 0.48598130841121495), ('Contrib', 0.653250773993808), ('Method', 0.3448275862068966), ('Conc', 0.2867647058823529), ('TOTAL_ACCURACY', 0.5160256410256411)])
    print(evaluator.build_confusion_matrix())
    print(evaluator.exact_match_accuracy())
