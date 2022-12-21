from collections import defaultdict
from pathlib import Path
from typing import Union, Optional
from collections.abc import Sequence

from numpy.typing import NDArray
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

import cv
from data_processing import *
import data_processing as dp
import metrics
import models


DATADIR = Path("data")

#########################################################


def rmse(x, y):
    return np.sqrt(((x - y) ** 2).mean())


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
    add_if(
        "phone",
        load_passive_phone,
        "df_passive_phone_communication_features_brighten_v2.csv",
    )
    add_if(
        "weather", load_passive_weather, "df_passive_weather_features_brighten_v2.csv"
    )
    # TODO: Maybe add weather data too

    return dailies


# feature_selection_jcompat:
# If TRUE, feature_selection will act as in J's original code, which means:
# It only works for N_SPLIT == 1, at the same time as standard features, and has plots
# If FALSE, will simply train a random forest, but also use feature selection before
# predicting results.


def train(*args, **kwargs):
    """Thin wrapper on top of train_cv for training without cross-validation"""
    return train_cv(*args, **kwargs, N_SPLIT=1)


def train_cv(
    MODEL_TYPE: str = "random-forest",
    TYPE: str = "classification",
    TARGET: str = "value",
    SPLIT_BY_PARTICIPANT: bool = False,
    TEST_TAKE_FIRST: int = 0,
    SEED: int = 550,
    N_SPLIT: int = 5,
    return_csv: bool = False,
    feature_selection: bool = False,
    feature_selection_jcompat: bool = False,
    partition: Optional[dict[int, int]] = None,
    verbose: bool = False,
    plot: bool = False,
    dailies_names: Optional[Sequence[str]] = ("locations",),
    test_size: Union[float, int] = 0.15,
    aggregate: bool = False,
    model_kwargs: Optional[dict] = None,
    use_demographics: bool = True,
    dir: Union[str, Path] = DATADIR,
):
    """
    Run an experiment optionally using cross-validation given model and data properties.

    A massive monolithic function encapsulating all our model/data/experiment combinations
    for easy trials and reproduction.

    Parameters
    ----------
    MODEL_TYPE : str
        'random-forest' or 'rnn'. Selects a backend model from models/
        See in the function definition model_type2cls for their arguments.
    TYPE : str
        Type of dataset: 'classification' or 'regression'. 'regression' is only supported
        for 'random-forest' models.
    TARGET : str
        'value' or 'diff'. value implies predicting the target class/value directly, while
        diff implies predicting its change.
    SPLIT_BY_PARTICIPANT : bool
        If specified, data will be split by participant, which is how it should ideally be.
        Data in the train and test sets will be from separate participants, ignoring
        TEST_TAKE_FIRST.

        False implies rows will be split randomly, e.g. say one patient's rows are r0,r1,r2,r3:
        r0,r2,r3 could end up in the training set and r1 in the test set.
    TEST_TAKE_FIRST : int
        If greater than zero and SPLIT_BY_PARTICIPANT=True, the first TEST_TAKE_ROWS rows from
        each participant in the test set will be removed from the test set and added to the
        training set. Good for having "personalized" models.
    SEED : int
        Random seed for reproducibility.
    N_SPLIT : int
        Number of splits for cross-validation. We use 10 for most experiments. Special cases:
            1: No cross-validation. A train,test split will be used instead.
            0: Leave-one-out cross-validation. A little extreme and high-variance...
    return_csv : bool
        If specified, return the loaded and prepared CSV data instead of doing anything else.
    feature_selection : bool
        If specified when MODEL_TYPE == 'random-forest', feature selection will be applied to
        the output model and the model will be retrained.
    feature_selection_jcompat : bool
        If specified, feature_selection will have more complex behavior such as plotting and
        returning multiple metrics on top of non-feature selection results, as in J's original
        implementation.
    partition : Optional[dict[int, int]]
        If specified as a dict mapping 0-27 to ints, will be used to map PHQ9 sum values to
        classes. e.g. 0-10: 0, 11-20: 1, 21-27: 2 could be one partition.
    verbose : bool
        If specified, print a lot of stuff. Otherwise, print less stuff.
    plot : bool
        If specified, produce plots in some helper functions.
    dailies_names : Optional[Sequence[str]]
        Names of dailies to be loaded as data for the input. 'locations', 'mobility' and
        'phone' are possibilities, as well as an empty list or None for no daily data.
    test_size : Union[float, int]
        Passed to sklearn.model_selection.train_test_split when not using cross-validation,
        i.e. N_SPLIT == 1
    aggregate: bool
        If specified, directly return the statistics of the metrics with *_mean and *_std keys,
        instead of an array containing the result for each split.
    model_kwargs: Optional[dict]
        If specified, arguments to be passed to the model constructor. Otherwise, some default
        arguments will be used.
    use_demographics: bool
        If specified as False, demographics information will be removed from the data.
    dir: Union[str, Path]
        Specifies the directory containing the csv data

    Returns
    ------
    In the usual cross-validation case, a dictionary mapping metrics to numpy arrays containing
    the metric result from each fold.

    When N_SPLIT == 1, is more complicated and depends on other options, mainly for backward
    compatibility.
    """

    # TODO: Ideally, should use the logging module, and set filter levels externally
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if dailies_names is None:
        dailies_names = []

    # loading and merging the data
    dir = Path(dir)
    phq9 = dp.load_phq9_targets(
        dir / "df_phq9.csv", type=TYPE, target=TARGET, partition=partition
    )
    demographics = load_demographics(dir / "df_demographics.csv")
    dailies = load_dailies(*dailies_names, dir=dir)

    combined, _ = dp.combine(  # this also caches the loaded csv with unique name
        phq9,
        dailies=dailies,
        constants=[demographics] if use_demographics else [],
        prev_phq9=False,
        verbose=verbose,
    )

    combined = combined.reset_index()

    model_type2cls = {
        "random-forest": (
            models.SKLearnRandomForest,
            {"n_estimators": 300, "n_jobs": -1, "random_state": SEED},
        ),
        "rnn": (models.LitRNNModel, {"hidden_size": 128, "num_layers": 2}),
        "mlp": (models.LitMLPModel, {"hidden_size": 128}),
        "xgboost": (models.XGBClassifier, {"n_estimators": 100, "reg_lambda": 1e-2}),
        "mostfreq": (models.MostFrequentPredictor, {}),
    }

    model_class, default_kwargs = model_type2cls[MODEL_TYPE]
    
    if model_kwargs is None:
        model_kwargs = default_kwargs

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

            model = RandomForestRegressor(
                n_estimators=300, n_jobs=-1, random_state=SEED
            )
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
            model.fit(x_train, y_train, x_test, y_test)

            # train_acc = 100 * model.score(x_train, y_train)
            # test_acc = 100 * model.score(x_test, y_test)
            # print(f'Train set accuracy: {train_acc:.2f}%')
            # print(f'Test set accuracy:  {test_acc:.2f}%')

            if plot:
                _, axes = plt.subplots(1, 2, figsize=(16, 6))
            else:
                axes = [None, None]

            if feature_selection:
                if MODEL_TYPE != "random-forest":
                    raise RuntimeError(
                        "feature_selection=True only supported for random-forest"
                    )

                forest = model.model

                if feature_selection_jcompat:  # Jstyle
                    train_score_sel, test_score_sel = feature_selection_results(
                        forest,
                        combined,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        verbose=verbose,
                    )

                    return train_mean, test_mean, train_score_sel, test_score_sel
                else:
                    sfm = SelectFromModel(forest, threshold=0.01, prefit=True)

                    # !!! WARNING: Serious hack replacing the original
                    # x_train and x_test, as well as the internal model of the SKLearnRandomForest
                    x_train = sfm.transform(x_train)
                    x_test = sfm.transform(x_test)
                    model = models.SKLearnRandomForest(existing_forest=sfm.estimator)
                    model.fit(x_train, y_train)

            vprint("Train set:")
            train_bal, train_mean = metrics.accuracy_info(
                y_train,
                model.predict(x_train),
                prefix="Training",
                ax=axes[0],
                plot=plot,
                verbose=verbose,
            )

            vprint("Test set:")
            test_bal, test_mean = metrics.accuracy_info(
                y_test,
                model.predict(x_test),
                prefix="Test",
                ax=axes[1],
                plot=plot,
                verbose=verbose,
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
        array_dict = { k: np.array(v) for k, v in metric_dict.items() }  # convert lists to arrays
        if not aggregate:
            return array_dict
        agg_dict = {}
        for k, v in array_dict.items():
            agg_dict[k + '_mean'] = v.mean()
            agg_dict[k + '_std'] = v.std()
        return agg_dict


def feature_selection_results(
    model, combined, x_train, y_train, x_test, y_test, verbose=True, bal=False
):
    feature_importances = model.feature_importances_

    sfm = SelectFromModel(model, threshold=0.01, prefit=True)
    x_train_new = sfm.transform(x_train)
    x_test_new = sfm.transform(x_test)
    sfm.estimator.fit(x_train_new, y_train)
    train_score_sel = 100 * sfm.estimator.score(x_train_new, y_train)
    test_score_sel = 100 * sfm.estimator.score(x_test_new, y_test)

    if verbose:
        print()
        print("After feature selection:")
        print("Train set shape:", x_train_new.shape)
        print("Test set shape:", x_test_new.shape)
        print()
        print(f"Train score: {train_score_sel:.2f}%")
        print(f"Test score:  {test_score_sel:.2f}%")

        combined = combined.drop(columns=["participant_id", "date", "target"])
        feature_importances = (
            pd.DataFrame(
                {"feature": combined.columns, "importance": feature_importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        plt.figure(figsize=(10, 10))
        sns.barplot(x="importance", y="feature", data=feature_importances)
        plt.show()

    if bal:
        train_bal_sel, train_mean_sel = metrics.accuracy_info(
            y_train, sfm.estimator.predict(x_train_new), plot=False, verbose=verbose
        )
        test_bal_sel, test_mean_sel = metrics.accuracy_info(
            y_test, sfm.estimator.predict(x_test_new), plot=False, verbose=verbose
        )
        return (
            train_score_sel,
            test_score_sel,
            train_bal_sel,
            train_mean_sel,
            test_bal_sel,
            test_mean_sel,
        )

    return train_score_sel, test_score_sel

PARTITIONS = {
    "P0": {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 3,
        17: 3,
        18: 3,
        19: 3,
        20: 3,
        21: 4,
        22: 4,
        23: 4,
        24: 4,
        25: 4,
        26: 4,
        27: 4,
    },
    "P1": {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 1,
        12: 1,
        13: 1,
        14: 1,
        15: 2,
        16: 2,
        17: 2,
        18: 2,
        19: 2,
        20: 2,
        21: 2,
        22: 2,
        23: 2,
        24: 2,
        25: 2,
        26: 2,
        27: 2,
    },
    "P2": {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 2,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 2,
        17: 2,
        18: 2,
        19: 2,
        20: 3,
        21: 3,
        22: 3,
        23: 3,
        24: 3,
        25: 3,
        26: 3,
        27: 3,
    },
    "P3": {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 2,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 3,
        16: 3,
        17: 3,
        18: 3,
        19: 3,
        20: 3,
        21: 3,
        22: 3,
        23: 3,
        24: 3,
        25: 3,
        26: 3,
        27: 3,
    },
    "P4": {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 3,
        11: 3,
        12: 3,
        13: 3,
        14: 3,
        15: 4,
        16: 4,
        17: 4,
        18: 4,
        19: 4,
        20: 5,
        21: 5,
        22: 5,
        23: 5,
        24: 5,
        25: 5,
        26: 5,
        27: 5,
    },
    "P5": {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 2,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 3,
        16: 3,
        17: 3,
        18: 3,
        19: 3,
        20: 4,
        21: 4,
        22: 4,
        23: 4,
        24: 4,
        25: 4,
        26: 4,
        27: 4,
    },
}
