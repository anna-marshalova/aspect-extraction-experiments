import os
import csv
from itertools import chain
from tqdm.autonotebook import tqdm
from typing import Tuple, List
from utils import ASPECTS_LIST, paths


class DataLoader:
    """Класс для загрузки данных из файлов"""
    def __init__(self, path_to_files: str = paths['data'], max_len: int = 100):
        """
        :param path_to_files: Путь к файлам с разметкой
        :param max_len: Максимальное количество токенов в одном тексте. Тексты, содержащие большее число токенов, делятся на несколько частей.
        """
        self._path_to_files = path_to_files
        self._max_len = max_len

    def _process_label(self, label: str) -> str:
        """
        Удаление аспектов, которые есть в датасете, но не участвуют в экспериментах
        :param label: Тэг из датасета
        :return: Тэг, состоящий только из тех аспектов, которые участвуют в экспериментах
        """
        tags = label.split('|')
        processed_label = [tag for tag in tags if tag in ASPECTS_LIST]
        if processed_label:
            return '|'.join(processed_label)
        else:
            return 'O'

    def process_file(self, file_path: str) -> Tuple[List, List]:
        """
          Загрузка одного файла с текстом. Если текст содержит большее заданного числа токенов, он делится на несколько частей.
          :param file_path: Путь к файлу
          :return: ``text_parts``:Части текста  (списки токенов),
                   ``text_part_labels``: Тэги для токенов в частях текста
          """
        with open(file_path, 'r', encoding='utf-8') as f:
            text_parts, text_part_labels = [], []
            tokens, token_labels = [], []
            # делим текст на части по предложениям, чтобы предложения не обрывались на середине
            sent, sent_labels = [], []
            reader = csv.DictReader(f)
            for row in reader:
                sent.append(row['token'])
                sent_labels.append(self._process_label(row['tag']))
                if row['token'] == '.':
                    # если с добавленным предложением текст не уместится в N=max_length токенов, переносим его в следующий семпл
                    if len(tokens) + len(sent) > self._max_len:
                        text_parts.append(tokens)
                        text_part_labels.append(token_labels)
                        tokens = []
                        token_labels = []
                    tokens.extend(sent)
                    token_labels.extend(sent_labels)
                    sent = []
                    sent_labels = []
            if len(tokens) > 0:
                text_parts.append(tokens)
                text_part_labels.append(token_labels)
            return text_parts, text_part_labels

    def load_dir(self, data_dir: str) -> Tuple[List[List], List[List]]:
        """
          Загрузка текстов из одной папки.
          :param data_dir: Путь к папке
          :return: ``samples``:Тексты  (списки токенов),
                   ``labels``: Тэги для токенов в текстах
          """
        samples = []
        labels = []
        filenames = sorted(os.listdir(data_dir))
        for filename in filenames:
            if filename.startswith('.'):
                continue
            file_path = os.path.join(data_dir, filename)
            text_parts, text_part_labels = self.process_file(file_path)
            samples.extend(text_parts)
            labels.extend(text_part_labels)
        return samples, labels

    def load_dataset(self, mode='cross_domain_flat') -> Tuple[List, List, List]:
        """
        Загрузка датасета
        :param mode: {'flat', 'cross_domain_flat','cross_domain'}
            ``flat``: Тексты загружаются из одной папки
            ``cross_domain_flat``: Тексты загружаются из разных папок, но сохраняются в одном списке
            ``cross_domain``: Тексты загружаются из разных папок и сохраняются в разных списках
        :return: ``dataset_samples``: Тексты
                ``dataset_labels``: Тэги для токенов в текстах
                ``domains``: Список доменов
        """
        if mode.startswith('cross_domain'):
            dataset_samples = []
            dataset_labels = []
            domains = sorted(os.listdir(self._path_to_files))
            for domain in tqdm(domains, desc='loading dataset'):
                domain_path = os.path.join(self._path_to_files, domain)
                samples, labels = self.load_dir(domain_path)
                dataset_samples.append(samples)
                dataset_labels.append(labels)
            if mode == 'cross_domain_flat':
                return list(chain.from_iterable(dataset_samples)), list(chain.from_iterable(
                    dataset_labels)), domains  # mode = 'cross_domain_flat' Tuple[List[List], List[List], List] Списки токенов в списках текстов
            return dataset_samples, dataset_labels, domains  # mode = 'cross_domain' Tuple[List[List[List]], List[List[List]], List] Списки токенов в списках текстов в списках доменов
        dataset_samples, dataset_labels = self.load_dir(self._path_to_files)
        domains = ['']
        return dataset_samples, dataset_labels, domains  # mode = 'flat' Tuple[List[List], List[List], ['']] Списки токенов в списках текстов
