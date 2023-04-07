import os
from typing import List, Any
from nltk.tokenize import wordpunct_tokenize

ASPECTS_LIST=['Task', 'Contrib', 'Method','Conc']
ROOT = '/content/drive/MyDrive/Маршалова Аня'
MAX_LENGTH_FOR_TOKENIZER = 200
paths = {'data': os.path.join(ROOT,'датасеты','cross_domain_bio_aspects'),
         'weights' : os.path.join(ROOT,'weights','AspectsWeights'),
         'results':(ROOT,'experiment_results','cross_domain')}
class2tag = dict(enumerate(ASPECTS_LIST))
tag2class=dict(zip(class2tag.values(),class2tag.keys()))
num_labels = len(tag2class.keys())

def flatten(nested_list:List[List[Any]]) -> List[Any]:
  return [item for nest in nested_list for item in nest]

def tokenize(text: str) -> List[str]:
    """
    Токенизация текста
    :param text: Текст
    :return: Список токенов
    """
    puncts = {'(', ')', ':', ';', ',', '.', '"', '»', '«', '[', ']', '{', '}','%','^'}

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