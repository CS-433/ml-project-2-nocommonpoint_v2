""" 
Contains generators for train-test splits; randomly or per-participant, with or without cross-validation.
"""

import numpy as np
import pandas as pd

import sklearn.model_selection as msel
from collections import Counter

import data_processing as dp
import models

DEFAULT_SPLIT_FN = models.SKLearnRandomForest.xy_split


def _make_group_indices(df):
    inds = df.groupby("participant_id").indices
    arrs = []
    i = 1
    for _, v in inds.items():
        arrs.append(i * np.ones(v.shape, dtype=np.int32))
        i += 1
    return np.concatenate(arrs)


def _xy_split(csv):
    y = csv["target"].to_numpy()
    x = csv.drop(
        csv.columns.intersection(["target", "participant_id", "date"]), axis=1
    ).to_numpy()
    return x, y


def _from_inds(x, y, train_i, test_i):
    try:
        return x[train_i], x[test_i], y[train_i], y[test_i]
    except IndexError:  # can happen if test set becomes empty in LeaveOneGroupOut CV
        return None, None, None, None


def _test_take_first(df, train_i, test_i, k):
    if k == 0:
        return train_i, test_i
    ctr = Counter()
    to_steal, to_keep = [], []
    for i in test_i:
        participant = df.participant_id[i]
        if ctr[participant] < k:
            to_steal.append(i)
            ctr[participant] += 1
        else:
            to_keep.append(i)
    return np.concatenate((train_i, np.array(to_steal))), np.array(to_keep)


### Single-split iterators


def per_row_once(df, test_size=0.15, split_fn=DEFAULT_SPLIT_FN):
    x, y = split_fn(df)
    yield msel.train_test_split(x, y, test_size=test_size, stratify=y)


def per_patient_once(df, test_size=0.15, test_take_first=0, split_fn=DEFAULT_SPLIT_FN):
    yield dp.train_test_split_participant(
        df, test_size, test_take_first, random_state=None, split_fn=split_fn
    )


### Cross validation split iterators


def per_row_cv(df, n_splits=5, split_fn=DEFAULT_SPLIT_FN):
    f = msel.StratifiedKFold(n_splits=n_splits)
    x, y = split_fn(df)
    for train_i, test_i in f.split(x, y):
        yield _from_inds(x, y, train_i, test_i)


def per_patient_cv(df, n_splits=5, test_take_first=0, split_fn=DEFAULT_SPLIT_FN):
    """
    Call this to create an x_train, x_test, y_train, y_test iterator with patient splitting.
    n_splits = None implies LeaveOneGroupOut, which has too high variance to be useful.
    """
    # assuming df contains the data points in order
    groups = _make_group_indices(df)
    if n_splits is None:
        f = msel.LeaveOneGroupOut()
    else:
        f = msel.StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    x, y = split_fn(df)
    if n_splits is None:
        print(f"# LeaveOneOut Splits: {f.get_n_splits(x, y, groups)}")
    for train_i, test_i in f.split(x, y, groups=groups):
        train_i, test_i = _test_take_first(df, train_i, test_i, test_take_first)
        yield _from_inds(x, y, train_i, test_i)
