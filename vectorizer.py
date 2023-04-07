import json
import numpy as np
from importlib import import_module
from typing import Tuple, List, Any
from utils import ASPECTS_LIST, MAX_LENGTH_FOR_TOKENIZER as MAX_LENGTH, tag2class, num_labels


class Vectorizer:

    def __init__(self, model_name: str):
        """
        :param model_name: Название модели
        """
        self.model_name = model_name
        with open('models.json', 'r', encoding='UTF-8') as js:
            models = json.load(js)
            model_config = models[self.model_name]
        tokenizer_class = getattr(import_module('transformers'), model_config['tokenizer_class'])
        self._tokenizer = tokenizer_class.from_pretrained(model_config["pretrained_model_name"], do_lower_case=False)
        self._tag2class = tag2class
        self._max_length = MAX_LENGTH

    def vectorize(self, text: List[str], token_labels: List[str], max_length: int = None) -> Tuple[
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
        tokenized_text, input_masks, labels = self._tokenize(text, token_labels, max_length=max_length)
        input_ids = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        tags = []
        for label in labels:
            tag = np.zeros(num_labels)
            label = label.split('|')
            for t in label:
                if t in ASPECTS_LIST:
                    tag[self._tag2class[t]] = 1.0
            tags.append(tag)
        input_ids = self._pad(input_ids, 0, max_length=max_length)
        input_masks = self._pad(input_masks, 0, max_length=max_length)
        tags = self._pad(tags, np.zeros(num_labels), max_length=max_length)

        return tokenized_text, input_ids, input_masks, tags

    def _pad(self, input: List[Any], padding: Any, max_length: int) -> List[Any]:
        """
        Доолняет список значениями до необходимой длины
        :param input: Список
        :param padding: Значение, которым дополнется список
        :param max_length: Длина, до которой нужно дополнить список
        :return: Дополненный список
        """
        if len(input) >= max_length:
            return input[:max_length]
        while len(input) < max_length:
            input.append(padding)
        return input

    def _tokenize(self, text: List[str], token_labels: List[str], max_length) -> Tuple[List[str], List[int], List[str]]:
        """
        Денение текста на bpe-токены и векторизация
        :param text: Текст (спсиок токенов)
        :param token_labels: Тэги для токенов в тексте
        :param max_length: Максимальное число bpe-токенов в векторизованном тексте (остальные обрезается)
        :return: ``tokenized_text``: Текст, разделенный на bpe-токены,
                 ``input_masks``: Маска для текста,
                 ``labels``:  Тэги для bpe-токенов в тексте
        """
        tokenized_text = []
        labels = []

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
