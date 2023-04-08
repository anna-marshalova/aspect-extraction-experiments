# Automatic aspect extraction from scientific texts in Russian

- This project is part of the [TERMinator (Tool for Information extraction from Russian texts)](https://github.com/iis-research-team/terminator).
- [Here](https://github.com/iis-research-team/ruserrc-dataset/tree/master/ruserrc_aspects) you can find the dataset, on which the models were trained.

If you find this repository useful, feel free to cite this paper:
`Marshalova A.E., Bruches E.P., Batura T.V. [Aspect extraction from scientific paper texts](http://swsys.ru/files/2022-4/698-706.pdf). Software
& Systems, 2022, vol. 35, no. 4, pp. 698–706 (in Russ.). DOI: 10.15827/0236-235X.140.698-706.`

Module description:
- `data_loader.py`: loads dataset from csv-files 
- `model.py`: creates a model
- `vectorizer.py`: vectorizes texts and labels
- `trainer.py`: used for training the model
- `predictor.py`: extracts aspects from texts
- `evaluator.py`: counts metrics for predictions
- `tag annotator.py`: annotates texts with predicteg aspects
- `extractor.py`: outputs extracted aspects and normalizes them
- `utils.py`: contains some constants, variables and functions 
- `models.json`: contains configs for different models
- `train_config.json`: contains training configurations
