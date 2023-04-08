import os
import csv
from typing import List, Union
from tqdm.autonotebook import tqdm
from IPython.display import HTML, display_html
from utils import ROOT, paths
from predictor import Predictor


class TagAnnotator:
    """Класс для автоматической разметки текстов тэгами аспектов"""

    def __init__(self, predictor: Predictor):
        """
        :param predictor: Объект класса для получения предсказаний модели
        """
        self.predictor = predictor
        #цвета, которыми будут выделяться аспекты
        self._ASPECT2COLOR = {'Task': '#A5C033', 'Contrib': '#DA95B8', 'Method': '#6EC4DB', 'Conc': '#F5B527'}
        self._css_aspects = ' '.join(
            [f'{aspect} {{background-color:{color}}}' for aspect, color in self._ASPECT2COLOR.items()])
        self._html_legend = ' '.join(
            [f'<{aspect} class="legend">{aspect}</{aspect}>' for aspect in self._ASPECT2COLOR.keys()])

    def annotate_with_tags(self, text: Union[List[str], str], labels: List[str] = None) -> str:
        """
        Разметка текста тэгами аспектов
        :param text: Текст (токенизированный или нет)
        :param labels: Список тэгов (нужен, например, для выполнения разметки известными тэгами из датасета)
        :return: Текст, размеченный тэгами аспектамов.
        Пример: В статье <Contrib> предложен инструмент для <Task> распознавания речи </Task> </Contrib>. ...
        """
        # TODO longer example
        text_with_tags = []
        prev_label = 'O'
        if labels:
            assert type(text) == list, "Only tokenized text can be annotated with provided labels"
            assert len(text) == len(labels), f"Cannot annotate {len(text)} tokens with {len(labels)} labels "
            result = list(zip(text, labels))
        else:
            result = self.predictor.extract(text)
        for i, (cur_token, cur_label) in enumerate(result):
            if i >= 1:
                _, prev_label = result[i - 1]
            # аспект продолжается (тэги не ставим)
            if cur_label == prev_label:
                text_with_tags.append(cur_token)
            else:
                cur_tags = cur_label.split('|')
                prev_tags = prev_label.split('|')
                # списки начинающихся и закончившихся аспектов
                open = [tag for tag in cur_tags if tag not in prev_tags]
                close = [tag for tag in prev_tags if tag not in cur_tags]
                # формируем открывающие и закрывающие тэги
                open_tag = [f'<{tag}> ' for tag in open if tag != 'O']
                close_tag = [f'</{tag}>' for tag in reversed(close) if tag != 'O']
                text_with_tags.extend(close_tag)
                text_with_tags.extend(open_tag)
                text_with_tags.append(cur_token)
        return ' '.join(text_with_tags)

    def annotate_with_colors(self, text: Union[List[str], str], padding: str = '20px', width: str = '50%', **kwargs) -> str:
        """
        Создание HTML разметка в формате строки, в которой каждый аспект выделен определенным цветом
        :param text: Текст (токенизированный или нет)
        :param padding: Размер отступов между элементами легенды в html
        :param width: Ширина страницы с разметкой
        :return: HTML разметка в формате строки.
        Содержит текст, в котором каждый аспект выделен определенным цветом и легенду, позволяющую понять, каким цветом обозначается каждый аспект.
        """
        css_body = f'body {{color:black; background-color:white;width:{width};padding:{padding};}} .legend {{padding: 2px;}} #legend-wrapper {{margin-bottom:10px;margin-top:10px}}'
        annot = self.annotate_with_tags(text, labels=labels).replace(' </', '</').replace('></', '> </')
        html = f'<style>{css_body} {self._css_aspects} </style> <div id="legend-wrapper"> {self._html_legend} </div> <div>{annot}</div>'
        return html

    def display_annotation_with_color(self, text: Union[List[str], str], **kwargs):
        """
        Вывод на экран текста, в котором каждый аспект выделен определенным цветом
        :param text: Текст (токенизированный или нет)
        """
        html = self.annotate_with_colors(text, **kwargs)
        display_html(HTML(html))

    def annotate_csv(self, texts: List[Union[List[str], str]], annot_dir: str = paths['examples'],
                     filename: str = 'test_auto_annotated.csv'):
        """
        Разметка текстов тэгами аспектов и запись разметки в таблицу в формате csv
        :param texts: Тексты (токенизированные или нет)
        :param annot_dir: Путь к папке для сохранения разметки
        :param filename: Имя файла, в которой сохранится разметка
        """
        fields = ['id', 'text']
        path = os.path.join(ROOT, annot_dir, filename)
        with open(path, 'w', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for text_id, text in enumerate(tqdm(texts)):
                text_with_labels = self.annotate_with_tags(text)
                writer.writerow({'id': text_id, 'text': text_with_labels})
