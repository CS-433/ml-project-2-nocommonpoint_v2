# CS-433 Project: Predicting Depression from Passive Phone Data

This repository contains our (team NoCommonPoint) code and report for our project attempting to predict depression levels on the BrightenV2 dataset. We were supervised by Dr. Giulia Da Poian and Cristina Gallego-Vazquez from the [Sensory-Motor Systems Lab](https://sms.hest.ethz.ch/) in ETH Zurich.

## Environment Reference

To be able to run our code and reproduce our experiments, Python 3.10 and a bunch of libraries including pytorch are necessary. For platform-independence, we stored the library names with no specific version in `requirements.txt`, which makes it easy to create a suitable environment using [Anaconda](https://www.anaconda.com/):

```sh
conda create -n nocommonenv python=3.10
conda activate nocommonenv
conda install --file requirements.txt -c conda-forge -c pytorch
```

## Our Reproduction Notebooks

The notebooks for reproducing the numerical figures and tables we put in our project report and containing extra insights can be found under `plots/`. 

* `exploration_overfitting.ipynb`: Contains code to reproduce our data exploration plots/info along with regularization attempts.
* `clean-csv.ipynb`: The initial notebook we used to clean up the original CSV's under `original_data`.

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

