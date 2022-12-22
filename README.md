# CS-433 Project: Predicting Depression from Passive Phone Data

This repository contains our (team NoCommonPoint) code and report for our project attempting to predict depression levels on the BrightenV2 dataset. We were supervised by Dr. Giulia Da Poian and Cristina Gallego-Vazquez from the [Sensory-Motor Systems Lab](https://sms.hest.ethz.ch/) in ETH Zurich.

## Environment Reference

To be able to run our code and reproduce our experiments, Python 3.10 and a bunch of libraries including pytorch are necessary. For platform-independence, we stored the library names with no specific version in `requirements.txt`, which makes it easy to create a suitable environment using [Anaconda](https://www.anaconda.com/):

```sh
conda create -n nocommonenv python=3.10
conda activate nocommonenv
conda install --file requirements.txt -c conda-forge -c pytorch
```

## Training a Simple Model

Training a model does not take long and is as easy as calling the train function! For example, to train a random forest model with 10-fold cross-validation, location and mobility features and k = 1:

```sh
$ python -i train.py
>>> train_cv(MODEL_TYPE='random-forest', TEST_TAKE_FIRST=1, SPLIT_BY_PARTICIPANT=True, 
             dailies_names=['locations', 'mobility'], N_SPLIT=10, aggregate=True)
{'train_bal_mean': 0.9535924165933917, 'train_bal_std': 0.002506950100629083, 
'train_mean_mean': 0.9495050520572124, 'train_mean_std': 0.0027495540000864057, 
'test_bal_mean': 0.3016172246722251, 'test_bal_std': 0.07304939578039911, 
'test_mean_mean': 0.3638339057429249, 'test_mean_std': 0.06404656136461256}
```

The returned value is a dictionary containing the mean/std of metrics over all folds. For our experiments, check out the notebooks explained below.

## Our Reproduction Notebooks

The notebooks for reproducing the numerical figures and tables we put in our project report and containing extra insights are listed here: 

* `exploration_overfitting.ipynb`: Contains code to reproduce our data exploration plots/info along with regularization attempts.
* `plots/plots.ipynb`: Generating the data for different feature choices and different target classes.
* `feateng.ipynb`: Experiments and results for feature engineering.
* `Models.ipynb`: A small notebook training with different models (random forest, XGBoost, MLP, RNN) and comparing their results, supplementary to the report.


We also have some notebooks we used during development that are not directly relevant to the report under `dev/` for the sake of completeness, they can be moved to the root directory and ran:
* `clean-csv.ipynb`: The initial notebook we used to clean up the original CSV's under `original_data/` and transformed them to the ones under `data/`.
* `w12.ipynb` and `w13.ipynb`: Extensive experiments with different target classes.
* `feature_selection.ipynb`: Feature selection on random forests, not included in the report since it had a small effect.
* `Foresting.ipynb`: Contains many initial experiments, including class distributions, per-class accuracies etc.ß
* `EDA.ipynb`: Initial data exploration.

## Summary of Our Project Layout/Code Files

* `train.py`: Contains the absolute monolith functions `train` and `train_cv` which can encapsulate all our experiments with 20+ parameters. Probably the only ones to be used directly.
* `cv.py`: Contains our code for preparing single/cross-validation datasets based on different splitting methods, such as keeping patients in training and test sets separate.
* `data_processing.py`: Includes the code used for loading different CSV files and pre-processing them to prepare the data for training.
* `metrics.py`: Contains some simple metrics for measuring model performance.
* `models`: Contains a file for each model type conforming to our model interface.
    * `protocol.py`: Contains the interface for our models.
    * `random_forest.py`: Our bread and butter, the scikit-learn random forest model.
    * `xgboost.py`: An XGBoost model as an alternative to random forests. 
    * `mlp.py`: A *shocking* multi-layer perceptron using `pytorch-lightning`
    * `rnn.py`: Similarly, an RNN using `pytorch-lightning`
    * `dummy.py`: A dummy model always predicting the most frequent class
    * `const_participant.py`: A smarter dummy model always predicting the last class seen for each participant

