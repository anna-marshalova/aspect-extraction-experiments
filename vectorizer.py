import json
import numpy as np
from importlib import import_module
from typing import Tuple, List, Any, Union
from utils import MAX_LENGTH_FOR_TOKENIZER as MAX_LENGTH, ASPECTS_LIST, paths, tag2class, num_labels
from config import get_model_config

class Vectorizer:
    """Класс для векторизации текстов и тэгов"""

    def __init__(self, model_name: str):
        """
        :param model_name: Название модели
        """
        self._model_name = model_name
        self._model_config = get_model_config(self._model_name)
        self._tokenizer = self._model_config.transformer_tokenizer.from_pretrained(self._model_config.pretrained_model_name, do_lower_case=False)
        self._max_length = MAX_LENGTH
        self._tag2class = tag2class

    def vectorize(self, text: List[str], token_labels: List[str] = None, max_length: int = None) -> Tuple[
        List[str], List[int], List[int], List[int]]:
        """
        Векторизация текста и тэгов токенов в тексте
        :param text: Текст (список токенов)
        :param token_labels: Список тэгов для токенов из текста
        :param max_length: Максимальное число bpe-токенов в векторизованном тексте (остальные обрезается)
        :return:   ``tokenized_text``: Текст, разделенный на bpe-токены,
                   ``input_ids``: Вектор текста,
                   ``input_masks``: Маска для текста,
                   ``tags``: One-hot-encoded тэги для токенов в тексте
        """
        if not max_length:
            max_length = self._max_length
        if not token_labels:
            token_labels = ['O']*len(text)
        tokenized_text, input_masks, labels = self._tokenize(text, token_labels, max_length=max_length)
        input_ids = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        tags = []
        for label in labels:
            tags.append(self.vectorize_label(label))
        input_ids = self._pad(input_ids, 0, max_length=max_length)
        input_masks = self._pad(input_masks, 0, max_length=max_length)
        tags = self._pad(tags, np.zeros(num_labels), max_length=max_length)

        return tokenized_text, input_ids, input_masks, tags

    def vectorize_label(self, label:str) -> np.array:
        """
        Преобразует тэг в one-hot вектор
        :param label: Тэг
        :return: One-hot вектор для тэга
        """
        vector = np.zeros(num_labels)
        classes = [aspect in label.split('|') for aspect in ASPECTS_LIST]
        vector[classes] = 1
        return vector

    def _pad(self, input: List[Any], padding: Any, max_length: int) -> List[Any]:
        """
        Доолняет список значениями до необходимой длины
        :param input: Список
        :param padding: Значение, которым дополняется список
        :param max_length: Длина, до которой нужно дополнить список
        :return: Дополненный список
        """
        if len(input) >= max_length:
            return input[:max_length]
        while len(input) < max_length:
            input.append(padding)
        return input

    def _tokenize(self, text: List[str], token_labels: List[str], max_length) -> Tuple[List[str], List[int], List[np.array]]:
        """
        Денение текста на bpe-токены и векторизация
        :param text: Текст (список токенов)
        :param token_labels: Тэги для токенов в тексте
        :param max_length: Максимальное число bpe-токенов в векторизованном тексте (остальные обрезается)
        :return: ``tokenized_text``: Текст, разделенный на bpe-токены,
                 ``input_masks``: Маска для текста,
                 ``labels``:  Тэги для bpe-токенов в тексте (one-hot encoded)
        """
        tokenized_text = []
        labels = []

        # делим слова на bpe-токены
        for token, label in zip(text, token_labels):
            tokenized_word = self._tokenizer.tokenize(token)
            n_subwords = len(tokenized_word)
            tokenized_text.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        inputs = self._tokenizer.encode_plus(
            tokenized_text,
            is_pretokenized=True,
            return_attention_mask=True,
            max_length=max_length,
            truncation=True)

        return tokenized_text, inputs['attention_mask'], labels

class SentVectorizer(Vectorizer):
    def __int__(self, model_name: str):
        """
        :param model_name: Название модели
        """
        super().__init__(model_name)

    def vectorize(self, text: Union[List[str], str], label: str = 'O', max_length: int = None) -> Tuple[
        Union[List[str], str], List[int], List[int], np.array]:
        """
        Векторизация текста
        :param text: Текст (строка или список токенов)
        :param label: Тэг
        :return:   ``text``: Текст в исходном виде,
                   ``input_ids``: Вектор текста,
                   ``input_masks``: Маска для текста,
                   ``tags``: One-hot-encoded тэг
        """
        input_ids, input_masks = self._tokenize(text)
        tag = self.vectorize_label(label)
        return text, input_ids, input_masks, tag

    def _tokenize(self, text: str) -> Tuple[List[int], List[int]]:
        """
        :param text: Текст (список токенов)
        :param text:
        :return: ``input_masks``: Маска для текста,
                 ``labels``:  Тэги для bpe-токенов в тексте (one-hot encoded)
        """
        inputs = self._tokenizer.encode_plus(
            text,
            return_attention_mask=True,
            max_length=self._max_length,
            padding='max_length',
            truncation=True)

        return inputs['input_ids'], inputs['attention_mask']