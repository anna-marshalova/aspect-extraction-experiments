import json
import logging
import numpy as np
from collections import Counter, defaultdict
from typing import Tuple, List, Union

from utils import MAX_LENGTH_FOR_TOKENIZER as MAX_LENGTH, class2tag, tokenize
from model import get_model
from vectorizer import Vectorizer
from heuristic_validator import HeuristicValidator


class Predictor:

    def __init__(self, model_name:str, threshold:float=0.5, weights_path:str=None, model=None):
        """
        :param threshold: Порог, по которому определяется принадлежность токена к аспекту
        :param weights_path: Путь к весам модели
        :param model_name: Название модели
        :param model: Модель
        """
        assert weights_path or model, 'Model or path to weights should be provided'
        self.model_name = model_name
        with open('models.json', 'r', encoding='UTF-8') as js:
            models = json.load(js)
            self.model_config = models[self.model_name]
        self.weights_path = weights_path
        if model:
            self._model = model
        else:
            self._model = get_model(model_name)
            self._model.load_weights(weights_path)
        self._vectorizer = Vectorizer(self.model_name)
        self._class2tag = class2tag
        self.threshold = threshold
        self._heuristic_validator = HeuristicValidator()

    def extract(self, text: Union[str, List[str]], use_heuristics:bool=True) -> List[Tuple[str, str]]:
        """ Извлечение аспектов из входного текста
        :param text: Входной текст, может быть строкой либо уже токенизированным (списком строк)
        :param use_heuristics: Применять ли к полученному результату эвристики
        :return: Список кортежей, в которых первый элемент - токен, второй элемент - тэги
        """
        if isinstance(text, str):
            tokens = tokenize(text)
        else:
            tokens = text
        labels = ['O' for i in range(len(tokens))]
        all_bpe_tokens = []
        all_predictions = []

        n_batches = int(len(tokens) / 50) + 1
        for i in range(n_batches):
            start = 50 * i
            end = 50 * i + 50
            if end > len(tokens):
                end = len(tokens)

            if start == end:
                break

            bpe_tokens, input_ids, input_masks, tags = self._vectorizer.vectorize(tokens[start: end], labels[start: end], max_length=MAX_LENGTH)
            preds = self._model.predict([np.array([input_ids]), np.array([input_masks])], verbose=False)[0]
            if self.model_config['model_class'].endswith("ForTokenClassification"): #у разных моделей разный формат выходов
                preds = preds[0]
            all_bpe_tokens.extend(bpe_tokens)
            all_predictions.extend(preds[:len(bpe_tokens)])

        result = self._get_preds_with_tokens(all_bpe_tokens, all_predictions)
        if isinstance(text, list):
            result = self._align_tokens(text, result)
        if use_heuristics:
            result = self._heuristic_validator.validate(result)
        return result

    def _align_tokens(self, input_tokens: List[str], result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Выравнивание токенов
        В случаях, когда на вход пришёл уже токенизированный текст, токены в результирующем списке могут отличаться от
        тех, что в исходном из-за bpe-токенизации. Поэтому нужно выровнять результирующий список относительно входного,
        т.е. список токенов в обоих списках должен совпадать
        :param input_tokens: Список токенов во входном списке
        :param result: Список кортежей, в которых первый элемент - токен, второй - тэг
        :return: Результат выравнивания - список кортежей, в которых первый элемент - токен, второй - тэг
        """
        resulted_tokens = [res[0] for res in result]
        if resulted_tokens == input_tokens:
            return result

        updated_result = []

        res_cursor = 0
        for i, token in enumerate(input_tokens):
            tokenized = tokenize(token)
            if len(tokenized) == 1:
                updated_result.append(result[res_cursor])
                res_cursor += 1
                continue

            full_resulted = []
            tags = Counter()
            for j in range(res_cursor, res_cursor + len(tokenized)):
                full_resulted.append(result[j][0])
                tags[result[j][1]] += 1

            tag = tags.most_common()[0][0]
            updated_result.append((''.join(full_resulted), tag))

            res_cursor += len(tokenized)

        if len(input_tokens) != len(updated_result):
            logging.warning(f'Alignment worked incorrect.{list(zip(input_tokens, updated_result))}')
        return updated_result

    def _get_preds_with_tokens(self, bpe_tokens:List[str], preds:List[str]) -> List[Tuple[str, str]]:
        """ Из предсказаний для bpe-токенов получаем предсказания для целых токенов
        :param bpe_tokens: Список bpe-токенов
        :param preds: Список предиктов от модели
        :return: Список кортежей, в которых первый элемент - полноценный токен, второй элемент - тэги
        """
        result = []
        token = []
        tags = []
        for bpe_token, pred in zip(bpe_tokens, preds):
            if bpe_token == '[UNK]':
                bpe_token = '–'
            if bpe_token.startswith('##'):
                token.append(bpe_token[2:])
                tags.extend(list(enumerate(pred)))
            else:
                if len(token) > 0:
                    self._process_token(result, tags, token)
                token = [bpe_token]
                tags = list(enumerate(pred))
        self._process_token(result, tags, token)
        return result

    def _process_token(self, result:List[str, str], tags:List[str], token:List[str]):
        """ Обработка токена: собираем его из bpe-токенов, выбираем нужные тэги
        :param result: Результирующий список с токенами и тэгами
        :param tags: Список тэгов, который был получен для составных bpe-токенов
        :param token: Список bpe-токенов для данного токена
        """
        token_str = ''.join(token)
        probability_sums = defaultdict(float)
        for tag, probability in tags:
            probability_sums[tag] += probability
        best_classes = [item[0] for item in sorted(probability_sums.items(), key=lambda item: item[1]) if
                        item[1] > self.threshold]
        best_tags = [self._class2tag[bc] for bc in best_classes]
        if len(best_tags) > 2:
            best_tags = best_tags[:2]
        if len(best_tags) == 0:
            best_tags = ['O']
        label = '|'.join(best_tags)
        result.append((token_str, label))

