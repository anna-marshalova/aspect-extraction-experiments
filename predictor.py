import os
import json
import logging
import numpy as np
from collections import Counter, defaultdict
from typing import Tuple, List, Union

from utils import MAX_LENGTH_FOR_TOKENIZER as MAX_LENGTH, ASPECTS_LIST, paths, class2tag, tokenize
from model import get_model
from vectorizer import Vectorizer
from heuristic_validator import HeuristicValidator
from config import get_model_config


class Predictor:
    """Класс для получения предсказаний модели"""

    def __init__(self, model_name:str, threshold:float=0.5, weights_dir:str=paths['weights'], weights_filename:str='weights.h5', model=None, vectorizer:Vectorizer=None):
        """
        :param threshold: Порог, по которому определяется принадлежность токена к аспекту
        :param weights_dir: Путь к папке с весами модели
        :param weights_filename: Название файла с весами модели
        :param model_name: Название модели
        :param model: Модель
        """
        assert weights_filename or model, 'Model or path to weights should be provided'
        self._model_name = model_name
        self.weights_path = os.path.join(weights_dir, weights_filename)
        if model:
            self._model = model
        else:
            self._model = get_model(model_name)
            self._model.load_weights(self.weights_path)
        self._model_config = get_model_config(self._model_name)
        if vectorizer:
            self._vectorizer = vectorizer
        else:
            self._vectorizer = Vectorizer(self._model_name)
        self._class2tag = class2tag
        self._threshold = threshold
        self._heuristic_validator = HeuristicValidator()

    def extract(self, text: Union[str, List[str]], use_heuristics:bool=True, **kwargs) -> List[Tuple[str, str]]:
        """ Извлечение аспектов из входного текста
        :param text: Входной текст, может быть строкой либо уже токенизированным (списком строк)
        :param use_heuristics: Применять ли к полученному результату эвристики
        :return: Список кортежей, в которых первый элемент - токен, второй элемент - тэги
        """
        if isinstance(text, str):
            tokens = tokenize(text)
        else:
            tokens = text
        all_bpe_tokens = []
        all_predictions = []

        # делим список токенов на батчи, которые будут последовательно обрабатываться
        n_batches = int(len(tokens) / 50) + 1
        for i in range(n_batches):
            start = 50 * i
            end = 50 * i + 50
            if end > len(tokens):
                end = len(tokens)

            if start == end:
                break

            bpe_tokens, input_ids, input_masks, _ = self._vectorizer.vectorize(tokens[start: end], max_length=MAX_LENGTH)
            preds = self._model.predict([np.array([input_ids]), np.array([input_masks])], verbose=False)[0]
            if self._model_config.transformer_model.__name__.endswith("ForTokenClassification"): #у разных моделей разный формат выходов
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

        # если списки токенов изначально совпадают, то сразу возвращаем результат
        resulted_tokens = [res[0] for res in result]
        if resulted_tokens == input_tokens:
            return result

        updated_result = []

        # фиксируем позицию токена в результирующем списке
        res_cursor = 0
        for i, token in enumerate(input_tokens):
            # токенизируем токен из входного списка. Если длина получившихся токенов == 1, то это не составной токен
            tokenized = tokenize(token)
            if len(tokenized) == 1:
                updated_result.append(result[res_cursor])
                res_cursor += 1
                continue

            full_resulted = []
            tags = Counter()
            # собираем все токены в результирующем списке, которые лежат в промежутке от res_cursor до
            # res_cursor + количество токенов в tokenized
            for j in range(res_cursor, res_cursor + len(tokenized)):
                full_resulted.append(result[j][0])
                tags[result[j][1]] += 1

            # на случай, если составным токенам были присвоены разные тэги, то выберем тэг с максимальной частотой
            tag = tags.most_common()[0][0]
            updated_result.append((''.join(full_resulted), tag))

            # переведём позицию курсора на количество составных частей исходного токена
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
            # если bpe-токен не является началом целого токена, то он начинается с "##"
            if bpe_token.startswith('##'):
                token.append(bpe_token[2:])
                tags.extend(list(enumerate(pred)))
            else:
                # если уже собрали токен до этого, то обработаем его и положим в результирущий список
                if len(token) > 0:
                    self._process_token(result, tags, token)
                token = [bpe_token]
                tags = list(enumerate(pred))
        # обработаем последний токен и положим его в результирующий список
        self._process_token(result, tags, token)
        return result

    def _process_token(self, result:List[Tuple[str, str]], tags:List[str], token:List[str]):
        """ Обработка токена: собираем его из bpe-токенов, выбираем нужные тэги
        :param result: Результирующий список с токенами и тэгами
        :param tags: Список тэгов, который был получен для составных bpe-токенов
        :param token: Список bpe-токенов для данного токена
        """
        # объединяем составные bpe-токены в единую строку
        token_str = ''.join(token)
        # складываем вероятности тэгов для каждого из составных bpe-токенов
        probability_sums = defaultdict(float)
        for tag, probability in tags:
            probability_sums[tag] += probability
        best_classes = [item[0] for item in sorted(probability_sums.items(), key=lambda item: item[1]) if
                        item[1] > self._threshold]
        best_tags = [self._class2tag[bc] for bc in best_classes]
        # если в списке кандидатов больше двух элементов, выбираются два тэга с наибольшими вероятностями
        if len(best_tags) > 2:
            best_tags = best_tags[:2]
        # если в список кандидатов пуст, выбирается тэг "O"
        if len(best_tags) == 0:
            best_tags = ['O']
        label = '|'.join(best_tags)
        result.append((token_str, label))

class SentPredictor(Predictor):
    def __init__(self, model_name: str, threshold: float = 0.5, weights_dir: str = paths['weights'],
                 weights_filename: str = 'weights.h5', model=None, vectorizer:Vectorizer=None):
        """
        :param threshold: Порог, по которому определяется принадлежность токена к аспекту
        :param weights_dir: Путь к папке с весами модели
        :param weights_filename: Название файла с весами модели
        :param model_name: Название модели
        :param model: Модель
        """
        super().__init__(model_name, threshold, weights_dir, weights_filename, model, vectorizer)
        self.aspects_array = np.array(ASPECTS_LIST)

    def extract(self, text: str) -> str:
        """
        Классификация предложения по аспектам
        :param text: Предложение в строковом формате
        :return: Аспекты в предложении, соединенные знаком |, либо 'O', если аспектов в предложении нет
        """
        _, input_ids, input_masks, _ = self._vectorizer.vectorize(text)
        preds = self._model.predict([np.array([input_ids]), np.array([input_masks])], verbose=False)[0]
        pred_labels = self.aspects_array[preds >= self._threshold]

        if len(pred_labels):
            return '|'.join(pred_labels)
        else:
            return 'O'