import os
from typing import List, Any

def flatten(nested_list:List[List[Any]]) -> List[Any]:
  return [item for nest in nested_list for item in nest]

ASPECTS_LIST=['Task', 'Contrib', 'Method','Conc']
ROOT = '/content/drive/MyDrive/Маршалова Аня'
MAX_LENGTH_FOR_TOKENIZER = 200
paths = {'data': os.path.join(ROOT,'датасеты','cross_domain_bio_aspects')}
class2tag = dict(enumerate(ASPECTS_LIST))
tag2class=dict(zip(class2tag.values(),class2tag.keys()))
num_labels = len(tag2class.keys())


