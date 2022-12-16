from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import cv
from data_processing import *
import data_processing as dp
import metrics
import models

DATADIR = Path("data")

def rmse(x, y):
    return np.sqrt(((x - y) ** 2).mean())

def train(*args, **kwargs):
    return train_cv(*args, **kwargs, N_SPLIT=1)

def load_dailies(*dailies_names, dir=DATADIR):
    dir = Path(dir)
    dailies = []

    # overcomplicating things with nested functions <3
    def add_if(name, loader, filename):
        if name in dailies_names:
            data = loader(dir / filename)
            dailies.append((name, data))
    
    add_if("locations", load_locations, "df_location_ratio.csv") 
    add_if("mobility", load_passive_mobility, "df_passive_mobility_features.csv")
    add_if("phone", load_passive_phone, 
           "df_passive_phone_communication_features_brighten_v2.csv")
    # TODO: Maybe add weather data too

    return dailies

def train_cv(
    MODEL_TYPE="random-forest",
    TYPE="classification",
    TARGET="value",
    SPLIT_BY_PARTICIPANT=False,
    TEST_TAKE_FIRST=0,
    SEED=550,
    N_SPLIT=5,
    return_csv=False,
    feature_selection=False,
    partition=None,
    verbose=False,
    plot=False,
    dailies_names=('locations',),
    test_size=0.15,
):
    # TODO: Ideally, should use the logging module, and set filter levels externally
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if dailies_names is None:
        dailies_names = []

    # loading and merging the data
    dir = DATADIR
    phq9 = dp.load_phq9_targets(dir / "df_phq9.csv", type=TYPE, target=TARGET)
    demographics = load_demographics(dir / "df_demographics.csv")
    dailies = load_dailies(*dailies_names, dir=dir)

    combined, _ = dp.combine( # this also caches the loaded csv with unique name
        phq9,
        dailies=dailies,
        constants=[demographics],
        prev_phq9=False,
    )

    model_type2cls = {
        "random-forest": (
            models.SKLearnRandomForest,
            {"n_estimators": 300, "n_jobs": -1, "random_state": SEED},
        ),
        "rnn": (models.LitRNNModel, {"hidden_size": 128, "num_layers": 2}),
    }

    model_class, model_kwargs = model_type2cls[MODEL_TYPE]

    combined = model_class.preprocess(combined)

    if return_csv:
        return combined

    # print('Train set shape:', x_train.shape)
    # print('Test set shape:', x_test.shape)
    # n = len(y_train) + len(y_test)
    # train_pct = len(y_train) / n * 100
    # test_pct = len(y_test) / n * 100
    # print(f'Ratio: {train_pct:.2f}%/{test_pct:.2f}%')
    # print()

    # Prepare the x_train, x_test, y_train, y_test generator depending on settings
    split_fn = model_class.xy_split
    if N_SPLIT == 1:
        gen = (
            cv.per_patient_once(combined, test_size, TEST_TAKE_FIRST, split_fn=split_fn) 
            if SPLIT_BY_PARTICIPANT
            else cv.per_row_once(combined, test_size, split_fn=split_fn)
        )
    else:
        gen = (
            cv.per_patient_cv(combined, N_SPLIT, TEST_TAKE_FIRST, split_fn=split_fn)
            if SPLIT_BY_PARTICIPANT
            else cv.per_row_cv(combined, N_SPLIT, split_fn=split_fn)
        )

    # Run training for each quadruple, usually cross-validation
    metric_dict = defaultdict(list)
    for i, (x_train, x_test, y_train, y_test) in enumerate(gen):
        if N_SPLIT != 1:
            vprint(f"--- Split {i} ---")

        # This can happen if using LeaveOneOut cross-validation, since taking
        # the first N rows may remove the train/test patient entirely
        # (None, None, None, None) is returned by the generator in that case
        if x_train is None:
            vprint("Skipped, because take first emptied the test set.")
            continue

        if TYPE == "regression":
            # Regression has only limited support for basic trials with random forests

            model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
            model.fit(x_train, y_train)

            train_rmse = rmse(y_train, model.predict(x_train))
            test_rmse = rmse(y_test, model.predict(x_test))

            train_score = model.score(x_train, y_train)
            test_score = model.score(x_test, y_test)

            metric_dict["train_rmse"].append(train_rmse)
            metric_dict["test_rmse"].append(test_rmse)
            metric_dict["train_score"].append(train_score)
            metric_dict["test_score"].append(test_score)
            
            vprint(f"Train set RMSE: {train_rmse:.4f}")
            vprint(f"Test set RMSE:  {test_rmse:.4f}")
            vprint(f"Train score:", train_score)
            vprint(f"Test score:", test_score)

        elif TYPE == "classification":

            model = model_class(**model_kwargs)
            model.fit(x_train, y_train)

            # train_acc = 100 * model.score(x_train, y_train)
            # test_acc = 100 * model.score(x_test, y_test)
            # print(f'Train set accuracy: {train_acc:.2f}%')
            # print(f'Test set accuracy:  {test_acc:.2f}%')

            if plot:
                _, axes = plt.subplots(1, 2, figsize=(16, 6))
            else:
                axes = [None, None]

            vprint("Train set:")
            train_bal, train_mean = metrics.accuracy_info(
                y_train,
                model.predict(x_train),
                prefix="Training",
                ax=axes[0],
                plot=plot,
                verbose=verbose
            )

            vprint("Test set:")
            test_bal, test_mean = metrics.accuracy_info(
                y_test, model.predict(x_test), prefix="Test", ax=axes[1], plot=plot, 
                verbose=verbose
            )

            metric_dict["train_bal"].append(train_bal)
            metric_dict["train_mean"].append(train_mean)
            metric_dict["test_bal"].append(test_bal)
            metric_dict["test_mean"].append(test_mean)

    # More special cases for backward compatibility with N_SPLIT == 1
    if N_SPLIT == 1:
        if TYPE == "regression":
            return metric_dict["train_score"][0], metric_dict["test_score"][0]
        elif TYPE == "classification":
            return metric_dict["train_bal"][0], metric_dict["test_bal"][0]
    else:
        return {k: np.array(v) for k, v in metric_dict.items()} # convert lists to arrays

