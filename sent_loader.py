import re
import os
import csv
import nltk
from typing import List, Tuple

from utils import ASPECTS_LIST, paths

nltk.download('punkt')


class SentLoader:

    def __init__(self, data_dir=paths['table_annotation'], aspects_list=ASPECTS_LIST, include_empty=True):
        """
        :param data_dir: Путь к папке с разметкой
        :param aspects_list: Список аспектов, участвующих в эксперименте
        :param include_empty: Брать ли из датасета предложения без аспектов
        """
        self._ASPECT_RE = re.compile(f'(</?[A-z]+?>)')
        self._data_dir = data_dir
        self._aspects_list = aspects_list
        self._include_empty = include_empty

    def load_texts_csv(self, filenames: List[str]) -> List[str]:
        """
        Загрузка размеченных текстов из разных файлов
        :param filenames: Список названий файлов в формате csv, с текстами, размеченными тэгами в колонке 'text'
        :return: Список размеченных текстов
        """
        texts = []
        for filename in filenames:
            with open(os.path.join(self._data_dir, filename), 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    texts.append(row['text'])
        return texts

    def extract_aspects_from_sent(self, sent: str) -> str:
        """
        Извлечение аспектов из предложения
        :param sent: Предложение
        :return: Строка, содержащая список тегов предложения, разделенных знаком |
        """
        tags = [aspect for aspect in self._aspects_list if f'<{aspect}>' in sent]
        if tags:
            return '|'.join(tags)
        return 'O'

    def load_dataset(self, filenames) -> Tuple[List[str], List[str]]:
        """
        Загрузка датасета
        :param filenames: Список названий файлов в формате csv, с текстами, размеченными тэгами в колонке 'text'
        :return: ``samples``:Список предложений (не токенизированных),
                 ``labels``: Список тэгов для каждого из предложений
        """
        texts = self.load_texts_csv(filenames)
        samples, labels = [], []
        for text in texts:
            sents = nltk.sent_tokenize(text)
            for sent in sents:
                label = self.extract_aspects_from_sent(sent)
                if label != 'O' or self._include_empty:
                    samples.append(self._ASPECT_RE.sub('', sent))
                    labels.append(label)
        return samples, labels
