# BugDetector
------------
This is a project to predict bugs and their priority, made from data in the Technical Debt Dataset.

## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── download_dataset.py
        |   └── preprocess_dataset.py
        │
        └── models         <- Scripts to train models and then use trained models to make predictions
            ├── predict_models.py
            └── train_models.py


## Download data
In order to train the models, we must have first the data correctly processed.
The first step is to download the data. To do so, run the `src/data/download_dataset.py` script from its directory. You can execute the following lines from the root of the project:
```bash
cd ./src/data/
python3 ./download_dataset.py
```
Once this is executed, a .zip file will be in `data/raw/raw-data.zip`. Please extract this file manually, as the project currently does not support automatic unzipping. The .csv files must be extracted into the `data/raw/` directory.


## Preprocess data
This step is necessary to transform the data into the final dataset, which will be used for training. In order to do so, run the `src/data/download_dataset.py` script from its directory. You can execute the following lines from the root of the project:
```bash
cd ./src/data/
python3 ./preprocess_dataset.py
```
This will automatically store the processed dataset into  `data/processed/bugs-multitarget.csv`.


## Data exploration
If you are interested in an analysis of the data, you can run the `notebooks/0-data-understanding.ipynb` notebook. Also, all plots made in this project have been saved into the `reports/figures` directory.


## Training
To train the models, run the `src/models/train_models.py` script from its directory. You can execute the following lines from the root of the project:
```bash
cd ./src/models/
python3 ./train_models.py
```
This will automatically store the 3 models (Decision Tree, MLP and Random Forest) into `models/`.


## Predicting
To predict the labels given input data, you can use the function `predict` in the `src/models/predict_models.py` script. The function parameter `model` accepts a string specifying the model that you want to use. The parameter `labels` is set to True if you want to receive the output labels directly, and the parameter `probs` is set to True if you want to receive the output probabilities. For instance, if we want to predict the output labels and probabilities using the three models, we would call `predict(input_X, model='all', labels=True, probs=True)`.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
