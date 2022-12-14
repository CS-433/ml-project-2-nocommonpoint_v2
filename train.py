import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import *
import data_processing as dp

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

DATADIR = Path("data")


def rmse(x, y):
    return np.sqrt(((x - y) ** 2).mean())


def train(
    TYPE="classification",
    TARGET="value",
    SPLIT_BY_PARTICIPANT=False,
    TEST_TAKE_FIRST=0,
    SEED=550,
    return_csv=False,
    feature_selection=False,
    partition=None,
    verbose=True,
    dailies_names=[],
):

    phq9 = dp.load_phq9_targets(
        DATADIR / "df_phq9.csv", type=TYPE, target=TARGET, partition=partition
    )
    locations = load_locations(DATADIR / "df_location_ratio.csv")
    demographics = load_demographics(DATADIR / "df_demographics.csv")

    dailies = [("locations", locations)]

    if "mobility" in dailies_names:
        mobility = load_passive_mobility(DATADIR / "df_passive_mobility_features.csv")
        dailies.append(("mobility", mobility))
    if "phone" in dailies_names:
        phone = load_passive_phone(
            DATADIR / "df_passive_phone_communication_features_brighten_v2.csv"
        )
        dailies.append(("phone", phone))

    # Print the first element for each element of dailies 
    
    for name, daily in dailies:
        print(name)

    print(dailies)
    combined, merge_result = dp.combine(
        phq9,
        dailies=dailies,
        constants=[demographics],
        prev_phq9=False,
    )
    combined = dp.rf_preprocess(combined)
    if return_csv:
        return combined

    if SPLIT_BY_PARTICIPANT:
        x_train, x_test, y_train, y_test = dp.train_test_split_participant(
            combined, 0.15, random_state=SEED, test_take_first=TEST_TAKE_FIRST
        )
    else:
        x, y = dp.xy_split(combined)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.15, random_state=SEED, stratify=y
        )

    n = len(y_train) + len(y_test)
    train_pct = len(y_train) / n * 100
    test_pct = len(y_test) / n * 100
    if verbose:
        print("Train set shape:", x_train.shape)
        print("Test set shape:", x_test.shape)
        print(f"Ratio: {train_pct:.2f}%/{test_pct:.2f}%")
        print()

    if TYPE == "regression":
        model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
    elif TYPE == "classification":
        model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    model.fit(x_train, y_train)

    train_rmse = rmse(y_train, model.predict(x_train))
    test_rmse = rmse(y_test, model.predict(x_test))
    train_score = 100 * model.score(x_train, y_train)
    test_score = 100 * model.score(x_test, y_test)
    if verbose:
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE:  {test_rmse:.4f}")
        print(f"Train score: {train_score:.2f}%")
        print(f"Test score:  {test_score:.2f}%")

    train_score_sel, test_score_sel = (
        feature_selection_results(
            model, combined, x_train, y_train, x_test, y_test, verbose=verbose
        )
        if feature_selection == True
        else (None, None)
    )

    return train_score, test_score, train_score_sel, test_score_sel


def feature_selection_results(
    model, combined, x_train, y_train, x_test, y_test, verbose=True
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

    return train_score_sel, test_score_sel
