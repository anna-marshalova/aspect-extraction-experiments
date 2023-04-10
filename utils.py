import os
from collections import Counter
from typing import List, Tuple
from nltk.tokenize import wordpunct_tokenize

ASPECTS_LIST = ['Task', 'Contrib', 'Method', 'Conc']
# ASPECTS_LIST = ['Contrib',  'Conc']
DOMAINS = ['biology_medicine', 'computer_science', 'history_philology', 'journalism', 'law', 'linguistics', 'math',
           'pedagogy', 'physics', 'psychology']
RANDOM_STATE = 42
ROOT = '/content/drive/MyDrive/Маршалова Аня'
MAX_LENGTH_FOR_TOKENIZER = 200
paths = {'data': os.path.join(ROOT, 'датасеты', 'cross_domain_bio_aspects'),
         'weights': os.path.join(ROOT, 'weights', 'AspectsWeights'),
         'results': os.path.join(ROOT, 'experiment_results', 'cross_domain'),
         'examples': os.path.join(ROOT, 'experiment_results', 'cross_domain', 'examples'),
         'cross_domain_weights': os.path.join(ROOT, 'weights', 'AspectsWeights', 'cross_domain'),
         'cross_domain_results': os.path.join(ROOT, 'experiment_results', 'cross_domain', 'cross_domain'),
         'table_annotation' : os.path.join(ROOT, 'разметка'),
         'model_config': 'models.json',
         'train_config': 'train_config.json'}
class2tag = dict(enumerate(ASPECTS_LIST))
tag2class = dict(zip(class2tag.values(), class2tag.keys()))
num_labels = len(tag2class.keys())


def sort_dataset(samples: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """
    Совместная сортировка текстов и тэгов по длине текстов
    :param samples: Тексты
    :param labels: Тэги
    :return: Список текстов, отсортированных по длине и список тэгов к ним
    """
    sorted_dataset = sorted(zip(samples, labels), key=lambda x: len(x[0]), reverse=True)
    sorted_samples = [sample for sample, label in sorted_dataset]
    sorted_labels = [label for sample, label in sorted_dataset]
    return sorted_samples, sorted_labels


def tokenize(text: str) -> List[str]:
    """
    Токенизация текста
    :param text: Текст
    :return: Список токенов
    """
    puncts = {'(', ')', ':', ';', ',', '.', '"', '»', '«', '[', ']', '{', '}', '%', '^'}
    tokens = wordpunct_tokenize(text)
    validated_tokens = []
    for token in tokens:
        is_all_puncts = True
        for char in token:
            if char not in puncts:
                is_all_puncts = False
        if is_all_puncts:
            validated_tokens.extend(list(token))
        else:
            validated_tokens.append(token)
    return validated_tokens


def count_labels(labels: List[str]) -> Counter:
    """Подсчет количества каждого тэга в разметке"""
    counter = Counter()
    for text in labels:
        for label in text:
            counter[label] += 1
    return counter
