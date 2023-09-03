# Automatic aspect extraction from scientific texts in Russian

- This project is part of the [TERMinator (Tool for Information extraction from Russian texts)](https://github.com/iis-research-team/terminator).
- This repository coinatins experiments and code for model training. Data and inference code can be found [here](https://github.com/anna-marshalova/automatic-aspect-extraction-from-scientific-texts).

## Examples

Example of automatic annotation (`tag_annotator.annotate_with_colors`)
<img alt = "Example of automatic annotation" src="auto_annot_57.png" width="500">

Abother example of automatic annotation (`extractor.print_extracted_aspects`)

```
ЗАДАЧА
1. Восстановление коэффициентов системы линейных разностных уравнений
ВКЛАД
1. Восстановление коэффициентов системы линейных разностных
2. Доказана сходимость и установлены оценки скорости сходимости метода
МЕТОД
1. Модификация метода обратной итерации
```

## Module description:
- `data_loader.py`: loads dataset from csv-files 
- `model.py`: creates a model
- `vectorizer.py`: vectorizes texts and labels
- `trainer.py`: used for training the model
- `predictor.py`: extracts aspects from texts
- `evaluator.py`: counts metrics for predictions
- `tag annotator.py`: annotates texts with predicteg aspects
- `aspect_extractor.py`: outputs extracted aspects and normalizes them
- `utils.py`: contains some constants, variables and functions 
- `models.json`: contains configs for different models
- `train_config.json`: contains training configurations
- `cross-domain-pipeline.json`: used for cross-domain experiments
