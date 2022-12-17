import os
import pickle
import hashlib
from collections import defaultdict

import pandas as pd
import numpy as np


# regression|classification & value|diff
def load_phq9_targets(
    phq9_path,
    type="classification",
    target="value",
    partition=None,
):
    csv = pd.read_csv(phq9_path, parse_dates=["phq9Date"])

    # TODO: derived sum is used in the csv right now

    #     if derived_sum:
    #         s = csv.loc[:, 'phq9_1':'phq9_9'].sum(axis=1)
    #     else:
    #         s = csv['phq9_sum']

    target_diff = False
    if target == "diff":
        target_diff = True

    if type == "regression":
        target_col = "phq9_sum_diff" if target_diff else "phq9_sum"
    elif type == "classification":
        target_col = "phq9_level_diff" if target_diff else "phq9_level"
        if not (target_diff) and partition is not None:
            target_col = "phq9_sum"
            csv[target_col] = csv[target_col].apply(lambda x: partition[x])
        else:
            csv[target_col] -= 1
    else:
        raise ValueError(f"Invalid PHQ9 target type {type}")

    csv.rename(columns={"phq9Date": "date", target_col: "target"}, inplace=True)

    csv.drop(
        csv.columns.difference(["participant_id", "date", "target"]),
        axis=1,
        inplace=True,
    )

    if target_diff:
        csv.dropna(axis=0, inplace=True)  # remove NaN rows, useful for target_diff

    return csv


def load_locations(loc_path):
    # Nothing superfluous. Nice!
    return pd.read_csv(loc_path, parse_dates=["date"])


def load_demographics(dem_path):
    csv = pd.read_csv(dem_path)

    # Remove some extra columns
    return csv.drop(["Unnamed: 0", "startdate", "study"], axis=1)


def load_passive_mobility(pm_path):
    csv = pd.read_csv(pm_path, parse_dates=["dt_passive"])

    csv.drop(["week"], axis=1, inplace=True)

    csv.came_to_work = csv.came_to_work.astype("float")

    csv.rename(columns={"dt_passive": "date"}, inplace=True)

    return csv


def load_passive_phone(pf_path):
    csv = pd.read_csv(pf_path, parse_dates=["date"])

    return csv.drop(["week"], axis=1)


def file_md5(path):
    """Calculate and return the MD5 sum of a file in a memory-inefficient way."""
    with open(path, "rb") as fp:
        return hashlib.md5(fp.read()).hexdigest()


def read_file(path):
    with open(path, "r") as fp:
        return fp.read().strip()


def write_file(path, string):
    with open(path, "w") as fp:
        fp.write(string)


def read_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def write_pickle(path, obj):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def gen_cached_paths(name):
    """Come up with a name for the cached version of the given CSV file."""
    names = (f"{name}_combined.pkl", f"{name}_merged.pkl", f"{name}_md5.txt")
    return [os.path.join("cache", f) for f in names]


def make_cached_name(dailies, constants, prev_phq9, daily_reduction):

    parts = []
    parts.append(f'dailies({",".join(n for n, _ in dailies)})')
    parts.append(f"constants({len(constants)})")  # TODO: Oof
    parts.append(f"prevphq9({prev_phq9})")
    parts.append(f'reduc({",".join(daily_reduction)})')

    return "_".join(parts)


def combine(
    phq9: pd.DataFrame,
    dailies=None,
    constants=None,
    prev_phq9=False,
    daily_reduction=["mean", "std"],
    verbose=False,
):
    """
    dailies: (name, csv) sequence, csv's are expected to have participant_id and date
    constants: (name, csv) sequence too, csv's are expected to have participant_id only
    """

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # How does this work?
    # Basically, "combine" daily data between multiple phq9 points.
    # example:
    # phq9  daily0  daily1
    # 3.00  718     333
    # 5.00  NaN     NaN
    # NaN   333     444
    # NaN   NaN     NaN
    # NaN   444     555
    # 4.00  NaN     NaN
    # becomes (with mean reduction):
    # phq9  daily0  daily1
    # 3.00  718     333
    # 5.00
    # 4.00  444     555

    if dailies is None:
        dailies = []

    if constants is None:
        constants = []

    # Loading up the cached version of the data, if it exists.
    md5 = None
    try:
        cached_name = make_cached_name(dailies, constants, prev_phq9, daily_reduction)
    except TypeError as e:
        cached_name = None
        vprint("Failed to create cached name, caching aborted.")
        vprint(e)
    cached_name = make_cached_name(dailies, constants, prev_phq9, daily_reduction)
    vprint("Cached name:", cached_name)
    # Check for cached version by name, load it if it exists.
    if cached_name is not None:
        os.makedirs("cache", exist_ok=True)
        cpath, mpath, hashpath = gen_cached_paths(cached_name)
        if os.path.exists(cpath):
            original_md5 = read_file(hashpath)
            # Check if the MD5 is the same as the MD5 of the provided file.
            md5 = file_md5(cpath)
            if original_md5 == md5:
                vprint(f"Found matching cached data at '{cpath}', loading from cache.")
                return read_pickle(cpath), read_pickle(mpath)
            else:
                vprint(
                    f"Cache exists at '{cpath}' but does not match, will be overwritten."
                )

    # Remember that PHQ9 is also daily.
    # Dailies should be (name, csv) pairs.

    # Also, let's add an extra column has_* to check for daily attributes later
    has_keys = ["has_phq9"]
    phq9["has_phq9"] = True

    for name, csv in dailies:
        k = f"has_{name}"
        csv[k] = True
        has_keys.append(k)

    # Now, merge everything. Use an 'outer' merge so that all entries are included,
    # and those who don't exist for that date get NaNs
    daily_merged = phq9
    for name, csv in dailies:
        daily_merged = daily_merged.merge(
            csv, on=["participant_id", "date"], how="outer"
        )
    daily_merged.sort_values(by=["participant_id", "date"], inplace=True)

    # To make things look a bit cleaner, replace NaN with False for has_* columns
    for has_key in has_keys:
        daily_merged[has_key].fillna(False, inplace=True)

    if prev_phq9:
        shifted_target = phq9.groupby("participant_id").apply(
            lambda x: x["target"].shift(1)
        )
        shifted_target.fillna(0, inplace=True)
        daily_merged["prev_target"] = shifted_target.reset_index(drop=True)

    # Now we need to do the reduction. Now, I'm sure there is some sort of

    # use a closure for grouping so we can cache some variables

    # note how to aggregate each column
    # we keep the last value for target and date (which is the phq9 date)
    # use the max for has_* keys so that it is 1 if at least one row is 1
    # the others are features, and we aggregate them with mean and std (daily_reduction)

    if dailies:

        def has_filt(col):
            return col.startswith("has_")

        def last_filt(col):
            return col in ("target", "date", "prev_target")

        def feature_filt(col):
            return col != "participant_id" and not (has_filt(col) or last_filt(col))

        cols = daily_merged.columns
        aggdict = {
            **{c: "last" for c in cols if last_filt(c)},
            **{c: "max" for c in cols if has_filt(c)},
            **{c: daily_reduction for c in cols if feature_filt(c)},
        }

        # now, build a rename dict to quickly rename columns
        rename_dict = {}
        for c in cols:
            c = str(c)
            if last_filt(c):
                rename_dict[c, "last"] = c
            elif has_filt(c):
                rename_dict[c, "max"] = c
            else:
                for reduct in daily_reduction:
                    rename_dict[c, reduct] = c + "_" + reduct

        # finally, the grouping closure
        def _group_and_reduce_phq9(df):
            inds = _group_by_phq9(df)
            aggframe = df.groupby(inds).agg(aggdict)
            aggframe.columns = [
                rename_dict[c] for c in aggframe.columns.to_flat_index()
            ]
            return aggframe

        pgrp = daily_merged.groupby("participant_id")
        daily_rows = pgrp.apply(_group_and_reduce_phq9)
    else:
        daily_rows = daily_merged  # no dailies to merge

    # Need to post_process the daily_rows a bit

    # 1. daily_rows has all the aggregated data we want, and each rows should have
    # a phq9 score. However, there is a possible issue: trailing location data
    # with no final phq9, which would lead to an empty group. We can remove such rows
    # easily by checking has_phq9, which should be 0 since no phq9 was aggregated.
    daily_rows = daily_rows[daily_rows.has_phq9 > 0.0]

    # 2. Replace NaN values with 0. Not great, but oh well...
    daily_rows.fillna(0.0, inplace=True)

    # Now, append the constants to each row
    for const in constants:
        daily_rows = daily_rows.merge(const, on="participant_id")

    # Add previous phq9 as context
    #     if prev_phq9:
    #         p = phq9.groupby('participant_id').apply(
    #             lambda x: pd.concat(
    #                 (x, x.rename({'target': 'prev_target'}, axis=1)['prev_target'].shift(1)), axis=1))

    if cached_name is not None:
        vprint("Caching generated data...")
        write_pickle(cpath, daily_rows)
        write_pickle(mpath, daily_merged)
        write_file(hashpath, file_md5(cpath))

    return daily_rows, daily_merged


def _group_by_phq9(df):
    # Assign a different group number to each consecutive phq9 'block'
    c = 0
    groups = []
    # Probably can derive this from some kind of cumsum, but brain is offline atm
    for k in df.has_phq9:
        groups.append(c)
        if k:
            c += 1
    return groups


def _aggregate_along(df):
    # Assign a different group number to each consecutive phq9 'block'
    c = 0
    groups = []
    # Probably can derive this from some kind of cumsum, but brain is offline atm
    for k in df.has_phq9:
        groups.append(c)
        if k:
            c += 1
    return groups


# These are the same as SKLearnRandomForest's, but are kept for backward compatibility


def rf_preprocess(df):
    target_columns = [c for c in df.columns if c.startswith("has_")]
    return df.drop(target_columns, axis=1)


def xy_split(csv):
    y = csv["target"].to_numpy()
    x = csv.drop(
        csv.columns.intersection(["target", "participant_id", "date"]), axis=1
    ).to_numpy()
    return x, y

def train_test_split_participant(
    csv, test_size=0.1, test_take_first=0, random_state=None, split_fn=xy_split
):
    participants = csv.participant_id.unique()
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(participants)
    cutoff = int(np.ceil(participants.size * test_size))

    train_csv = csv[csv.participant_id.isin(participants[cutoff:])]
    test_csv = csv[csv.participant_id.isin(participants[:cutoff])]

    if test_take_first > 0:
        #         if 'date' in test_csv.columns:
        #             test_csv.sort_values(by=['participant_id', 'date'], inplace=True)

        train_part = test_csv.groupby("participant_id").head(test_take_first)
        test_part = test_csv.groupby("participant_id").tail(-test_take_first)

        final_test_csv = test_part
        final_train_csv = pd.concat((train_csv, train_part))

        train_count = final_train_csv.participant_id.unique().size
        test_count = final_test_csv.participant_id.unique().size
        train_pct = 100 * train_count / (train_count + test_count)
        test_pct = 100 * test_count / (train_count + test_count)

        print(
            f"After test_take_first={test_take_first}, have {train_pct:.2f}% ({train_count}) "
            f"vs. {test_pct:.2f}% ({test_count}) participants in train/test set"
        )
    else:
        final_test_csv, final_train_csv = (test_csv, train_csv)

    x_train, y_train = split_fn(final_train_csv)
    x_test, y_test = split_fn(final_test_csv)

    return x_train, x_test, y_train, y_test






##### For compatibility with S's experiments

# regression|classification & value|diff
def load_phq9_each_targets(
    phq9_path,
    type="regression",
    target="value",
    sum_phq9=False,
    phq9_index=0,
    derived_sum=True,
):
    csv = pd.read_csv(phq9_path, parse_dates=["phq9Date"])

    # TODO: derived sum is used in the csv right now

    #     if derived_sum:
    #         s = csv.loc[:, 'phq9_1':'phq9_9'].sum(axis=1)
    #     else:
    #         s = csv['phq9_sum']
    phq9_targets_value = value_phq9_target(sum_phq9, phq9_index)
    target_diff = False
    if target == "diff":
        target_diff = True

    if type == "regression":
        target_col = "phq9_sum_diff" if target_diff else phq9_targets_value
    elif type == "classification":
        target_col = "phq9_level_diff" if target_diff else "phq9_level"
    else:
        raise ValueError(f"Invalid PHQ9 target type {type}")

    csv.rename(columns={"phq9Date": "date", target_col: "target"}, inplace=True)

    csv.drop(
        csv.columns.difference(["participant_id", "date", "target"]),
        axis=1,
        inplace=True,
    )

    if target_diff:
        csv.dropna(axis=0, inplace=True)  # remove NaN rows, useful for target_diff

    return csv


def value_phq9_target(sum_phq9, phq9_index):
    # the target is value
    # if sum_phq9 is true, use the sum provided by dataset
    # if phq9_index is not 0, the target is phq9_index (eg:phq9_3)
    # if phq_index is 0, target is sum of phq9 scores
    if (phq9_index > 0) & (phq9_index < 10):
        value_target = "phq9_" + str(phq9_index)
    elif phq9_index == 0:
        value_target = "sum_phq9" if sum_phq9 else "phq9_sum"
        print(value_target)
    else:
        raise ValueError(f"Invalid PHQ9 index {phq9_index}")
    return value_target


def load_baselinephq9(basephq9_path, type="regression"):
    csv = pd.read_csv(basephq9_path)

    # Remove some extra columns
    return csv.drop(["Unnamed: 0", "base_baselinePHQ9date", "study"], axis=1)


def load_phq2_targets(phq2_path, type="regression", target="value", phq2_index=0):
    # if phq2_index = 0, target is sum of phq2 questions, otherwise target is each phq2 questions
    csv = pd.read_csv(phq2_path)
    if (phq2_index == 1) | (phq2_index == 2):
        phq2_targets_value = "phq2_" + str(phq2_index)
    elif phq2_index == 0:
        phq2_targets_value = "phq2_sum"
    else:
        raise ValueError(f"Invalid PHQ2 index {phq2_index}")

    target_diff = False
    if target == "diff":
        target_diff = True

    if type == "regression":
        target_col = "phq2_sum_diff" if target_diff else phq2_targets_value
    else:
        raise ValueError(f"Invalid PHQ9 target type {type}")

    csv.rename(columns={target_col: "target"}, inplace=True)

    csv.drop(
        csv.columns.difference(["participant_id", "date", "target"]),
        axis=1,
        inplace=True,
    )

    if target_diff:
        csv.dropna(axis=0, inplace=True)  # remove NaN rows, useful for target_diff
    return csv

def combine_additively(
    phq9: pd.DataFrame,
    dailies=None,
    constants=None,
    prev_phq9=False,
    weighted=False,
    all_previous_data=False,
    previous_num=0,
    daily_reduction=["mean", "std"],
):
    """
    dailies: (name, csv) sequence, csv's are expected to have participant_id and date
    constants: (name, csv) sequence too, csv's are expected to have participant_id only
    weighted: weight the previous value inverse to the time length from last time point to current phq9 point
    previous_num: combine daily data from previous phq9 test. (1: last test, 2: last 2 tests)
    """

    # How does this work?
    # Basically, "combine" daily data phq9 points from the start
    # example:
    # phq9  daily0  daily1
    # 3.00  718     333
    # 5.00  NaN     NaN
    # NaN   333     444
    # NaN   NaN     NaN
    # NaN   444     555
    # 4.00  NaN     NaN
    # becomes (with mean reduction):
    # phq9  daily0  daily1
    # 3.00  718     333
    # 5.00  718     333
    # 4.00  498     444
    #

    if dailies is None:
        dailies = []

    if constants is None:
        constants = []

    # Remember that PHQ9 is also daily.
    # Dailies should be (name, csv) pairs.

    # Also, let's add an extra column has_* to check for daily attributes later
    has_keys = ["has_phq9"]
    phq9["has_phq9"] = True

    for name, csv in dailies:
        k = f"has_{name}"
        csv[k] = True
        has_keys.append(k)

    # Now, merge everything. Use an 'outer' merge so that all entries are included,
    # and those who don't exist for that date get NaNs
    daily_merged = phq9
    for name, csv in dailies:
        daily_merged = daily_merged.merge(
            csv, on=["participant_id", "date"], how="outer"
        )
    daily_merged.sort_values(by=["participant_id", "date"], inplace=True)

    # To make things look a bit cleaner, replace NaN with False for has_* columns
    for has_key in has_keys:
        daily_merged[has_key].fillna(False, inplace=True)

    if prev_phq9:  # previous phq9 test result
        shifted_target = phq9.groupby("participant_id").apply(
            lambda x: x["target"].shift(1)
        )
        shifted_target.fillna(0, inplace=True)
        daily_merged["prev_target"] = shifted_target.reset_index(drop=True)

    def has_filt(col):
        return col.startswith("has_")

    def last_filt(col):
        return col in ("target", "date", "prev_target")

    def feature_filt(col):
        return col != "participant_id" and not (has_filt(col) or last_filt(col))

    cols = daily_merged.columns
    aggdict = {
        **{c: "last" for c in cols if last_filt(c)},
        **{c: "max" for c in cols if has_filt(c)},
        **{c: daily_reduction for c in cols if feature_filt(c)},
    }

    # COLUMNS_INDEX = ['participant_id', 'date', 'target', 'prev_target', 'has_phq9']
    COLUMNS_INDEX = [c for c in cols if last_filt(c)]
    COLUMNS_INDEX.append("has_phq9")
    COLUMNS_INDEX.insert(0, "participant_id")
    COLUMNS_HAS = [c for c in cols if has_filt(c)]
    COLUMNS_HAS.pop(0)
    COUNT_COLUMNS = [c for c in cols if feature_filt(c)]

    rename_dict = {}
    for c in cols:
        if last_filt(c):
            rename_dict[c, "last"] = c
        elif has_filt(c):
            rename_dict[c, "max"] = c
        else:
            for reduct in daily_reduction:
                rename_dict[c, reduct] = str(c) + "_" + reduct

    # finally, the grouping closure
    def _group_and_reduce_phq9(df):
        inds = _group_by_phq9(df)
        aggframe = df.groupby(inds).agg(aggdict)
        aggframe.columns = [rename_dict[c] for c in aggframe.columns.to_flat_index()]
        return aggframe

    def _group_by_phq9(df):
        # Assign a different group number to each consecutive phq9 'block'
        c = 0
        groups = []
        # Probably can derive this from some kind of cumsum, but brain is offline atm
        for k in df.has_phq9:
            groups.append(c)
            if k:
                c += 1
        return groups

    def _reduce_phq9_cumulative(daily_merged, daily_reduction):
        # cumulative reduction from the start
        # TODO add more reduction methods, combine them within a dataframe
        for reduct in daily_reduction:
            if reduct == "mean":
                df_cumcount_t = daily_merged.copy()
                df_cumcount_t[COUNT_COLUMNS] = df_cumcount_t[COUNT_COLUMNS].notna()
                df_cumcount = (
                    df_cumcount_t.set_index(COLUMNS_INDEX)
                    .groupby(level=0)
                    .cumsum()
                    .reset_index()
                )

                df_cumsum = (
                    daily_merged.set_index(COLUMNS_INDEX)
                    .groupby(level=0)
                    .cumsum()
                    .reset_index()
                )
                df_cumsum.fillna(0, inplace=True)
                df_cummean = (
                    df_cumsum[COUNT_COLUMNS]
                    .div(df_cumcount[COUNT_COLUMNS])
                    .fillna(0)
                    .add_suffix("_mean")
                )
                df_cummax = (
                    daily_merged.set_index(COLUMNS_INDEX)
                    .groupby(level=0)
                    .cummax()
                    .reset_index()[COLUMNS_HAS]
                )

                df_cum = pd.concat(
                    [daily_merged[COLUMNS_INDEX], df_cummax, df_cummean], axis=1
                )

            elif reduct == "std":
                pass

            elif reduct == "max":
                df_cumcal = (
                    daily_merged.set_index(COLUMNS_INDEX)
                    .groupby(level=0)
                    .cummax()
                    .reset_index()[COUNT_COLUMNS]
                    .add_suffix("_max")
                )
                df_cummax = (
                    daily_merged.set_index(COLUMNS_INDEX)
                    .groupby(level=0)
                    .cummax()
                    .reset_index()[COLUMNS_HAS]
                )
                df_cum = pd.concat(
                    [daily_merged[COLUMNS_INDEX], df_cummax, df_cumcal], axis=1
                )

            elif reduct == "min":
                df_cumcal = (
                    daily_merged.set_index(COLUMNS_INDEX)
                    .groupby(level=0)
                    .cummin()
                    .reset_index()[COUNT_COLUMNS]
                    .add_suffix("_min")
                )
                df_cummax = (
                    daily_merged.set_index(COLUMNS_INDEX)
                    .groupby(level=0)
                    .cummax()
                    .reset_index()[COLUMNS_HAS]
                )
                df_cum = pd.concat(
                    [daily_merged[COLUMNS_INDEX], df_cummax, df_cumcal], axis=1
                )
            else:
                KeyError("reduction method not found")
        return df_cum

    # 1. daily_rows has all the aggregated data we want, and each rows should have
    # a phq9 score. However, there is a possible issue: trailing location data
    # with no final phq9, which would lead to an empty group. We can remove such rows
    # easily by checking has_phq9, which should be 0 since no phq9 was aggregated.
    if all_previous_data:
        daily_rows = _reduce_phq9_cumulative(daily_merged, daily_reduction)
        daily_rows.fillna(0.0, inplace=True)
        daily_rows = daily_rows[daily_rows.has_phq9 > 0.0]

    else:
        pgrp = daily_merged.groupby("participant_id")
        daily_rows = pgrp.apply(_group_and_reduce_phq9)
        daily_rows.fillna(0.0, inplace=True)
        daily_rows = daily_rows[daily_rows.has_phq9 > 0.0]
        daily_rows = daily_rows.reset_index()
        daily_rows.drop(columns=["level_1"], inplace=True)

        if previous_num == 0:
            pass
        else:
            if weighted:
                pass
            else:  # cumulative previous_num test data without weighting
                columns_cal = []
                for col in daily_rows.columns:
                    if col not in COLUMNS_INDEX and col not in COLUMNS_HAS:
                        columns_cal.append(col)

                daily_rows = (
                    daily_rows.set_index(COLUMNS_INDEX + COLUMNS_HAS)
                    .groupby(["participant_id"])[columns_cal]
                    .transform(lambda s: s.rolling(previous_num, min_periods=1).mean())
                    .reset_index()
                )

    # Now, append the constants to each row
    for const in constants:
        daily_rows = daily_rows.merge(const, on="participant_id")

    return daily_rows, daily_merged


def combined_additively(
    phq9,
    dailies=None,
    constants=None,
    prev_phq9=False,
    all_previous_data=False,
    previous_num=1,
    weighted=True,
    weighted_days=True,
    ewm_halflife="7 days",
    ewm_span=5,  # Specify decay in terms of span
    daily_reduction=["mean"],
):
    """
    dailies: (name, csv) sequence, csv's are expected to have participant_id and date
    constants: (name, csv) sequence too, csv's are expected to have participant_id only
    weighted: weight the previous value inverse to the time length from last time point to current phq9 point
    previous_num: combine daily data from previous phq9 test. (1: last test, 2: last 2 tests)
    """

    # How does this work?
    # Basically, "combine" daily data phq9 points from the start
    # example:
    # phq9  daily0  daily1
    # 3.00  718     333
    # 5.00  NaN     NaN
    # NaN   333     444
    # NaN   NaN     NaN
    # NaN   444     555
    # 4.00  NaN     NaN
    # becomes (with mean reduction):
    # phq9  daily0  daily1
    # 3.00  718     333
    # 5.00  718     333
    # 4.00  498     444
    #

    if dailies is None:
        dailies = []

    if constants is None:
        constants = []

    # Remember that PHQ9 is also daily.
    # Dailies should be (name, csv) pairs.

    # Also, let's add an extra column has_* to check for daily attributes later
    has_keys = ["has_phq9"]
    phq9["has_phq9"] = True

    for name, csv in dailies:
        k = f"has_{name}"
        csv[k] = True
        has_keys.append(k)

    # Now, merge everything. Use an 'outer' merge so that all entries are included,
    # and those who don't exist for that date get NaNs
    daily_merged = phq9
    for name, csv in dailies:
        daily_merged = daily_merged.merge(
            csv, on=["participant_id", "date"], how="outer"
        )
    daily_merged.sort_values(by=["participant_id", "date"], inplace=True)

    # To make things look a bit cleaner, replace NaN with False for has_* columns
    for has_key in has_keys:
        daily_merged[has_key].fillna(False, inplace=True)

    if prev_phq9:  # previous phq9 test result
        shifted_target = phq9.groupby("participant_id").apply(
            lambda x: x["target"].shift(1)
        )
        shifted_target.fillna(0, inplace=True)
        daily_merged["prev_target"] = shifted_target.reset_index(drop=True)

    def has_filt(col):
        return col.startswith("has_")

    def last_filt(col):
        return col in ("target", "date", "prev_target")

    def feature_filt(col):
        return col != "participant_id" and not (has_filt(col) or last_filt(col))

    cols = daily_merged.columns
    aggdict = {
        **{c: "last" for c in cols if last_filt(c)},
        **{c: "max" for c in cols if has_filt(c)},
        **{c: daily_reduction for c in cols if feature_filt(c)},
    }

    # COLUMNS_INDEX = ['participant_id', 'date', 'target', 'prev_target', 'has_phq9']
    COLUMNS_INDEX = [c for c in cols if last_filt(c)]
    COLUMNS_INDEX.append("has_phq9")
    COLUMNS_INDEX.insert(0, "participant_id")
    COLUMNS_HAS = [c for c in cols if has_filt(c)]
    COLUMNS_HAS.pop(0)
    COUNT_COLUMNS = [c for c in cols if feature_filt(c)]

    rename_dict = {}
    for c in cols:
        if last_filt(c):
            rename_dict[c, "last"] = c
        elif has_filt(c):
            rename_dict[c, "max"] = c
        else:
            for reduct in daily_reduction:
                rename_dict[c, reduct] = c + "_" + reduct

    # finally, the grouping closure
    def _group_and_reduce_phq9(df):
        inds = _group_by_phq9(df)
        aggframe = df.groupby(inds).agg(aggdict)
        aggframe.columns = [rename_dict[c] for c in aggframe.columns.to_flat_index()]
        return aggframe

    def _group_by_phq9(df):
        # Assign a different group number to each consecutive phq9 'block'
        c = 0
        groups = []
        # Probably can derive this from some kind of cumsum, but brain is offline atm
        for k in df.has_phq9:
            groups.append(c)
            if k:
                c += 1
        return groups

    def _reduce_phq9_cumulative(daily_merged, daily_reduction):
        # cumulative reduction from the start
        # TODO add more reduction methods, combine them within a dataframe
        df = daily_merged[COLUMNS_INDEX]
        for reduct in daily_reduction:
            match reduct:
                case "mean":
                    df_cummean = (
                        daily_merged.groupby(["participant_id"])[COUNT_COLUMNS]
                        .expanding()
                        .mean()
                        .add_suffix("_mean")
                        .reset_index()
                        .fillna(0)
                        .drop(columns=["level_1", "participant_id"])
                    )
                    df_cummax = (
                        daily_merged.set_index(COLUMNS_INDEX)
                        .groupby(level=0)
                        .cummax()
                        .reset_index()[COLUMNS_HAS]
                    )
                    df_cum = pd.concat(
                        [daily_merged[COLUMNS_INDEX], df_cummax, df_cummean], axis=1
                    )
                    df = df.merge(
                        df_cum,
                        left_on=COLUMNS_INDEX,
                        right_on=COLUMNS_INDEX,
                        how="left",
                    )

                case "std":
                    pass

                case "max":
                    df_cumcal = (
                        daily_merged.set_index(COLUMNS_INDEX)
                        .groupby(level=0)
                        .cummax()
                        .reset_index()[COUNT_COLUMNS]
                        .add_suffix("_max")
                    )
                    df_cummax = (
                        daily_merged.set_index(COLUMNS_INDEX)
                        .groupby(level=0)
                        .cummax()
                        .reset_index()[COLUMNS_HAS]
                    )
                    df_cum = pd.concat(
                        [daily_merged[COLUMNS_INDEX], df_cummax, df_cumcal], axis=1
                    )
                    df = df.merge(
                        df_cum,
                        left_on=COLUMNS_INDEX,
                        right_on=COLUMNS_INDEX,
                        how="left",
                    )

                case "min":
                    df_cumcal = (
                        daily_merged.set_index(COLUMNS_INDEX)
                        .groupby(level=0)
                        .cummin()
                        .reset_index()[COUNT_COLUMNS]
                        .add_suffix("_min")
                    )
                    df_cummax = (
                        daily_merged.set_index(COLUMNS_INDEX)
                        .groupby(level=0)
                        .cummax()
                        .reset_index()[COLUMNS_HAS]
                    )
                    df_cum = pd.concat(
                        [daily_merged[COLUMNS_INDEX], df_cummax, df_cumcal], axis=1
                    )
                    df = df.merge(
                        df_cum,
                        left_on=COLUMNS_INDEX,
                        right_on=COLUMNS_INDEX,
                        how="left",
                    )
                case _:
                    KeyError("reduction method not found")
        return df

    # 1. daily_rows has all the aggregated data we want, and each rows should have
    # a phq9 score. However, there is a possible issue: trailing location data
    # with no final phq9, which would lead to an empty group. We can remove such rows
    # easily by checking has_phq9, which should be 0 since no phq9 was aggregated.
    if all_previous_data:
        daily_rows = _reduce_phq9_cumulative(daily_merged, daily_reduction)
        daily_rows.fillna(0.0, inplace=True)
        daily_rows = daily_rows[daily_rows.has_phq9 > 0.0]

    else:
        pgrp = daily_merged.groupby("participant_id")
        daily_rows = pgrp.apply(_group_and_reduce_phq9)
        daily_rows.fillna(0.0, inplace=True)
        daily_rows = daily_rows[daily_rows.has_phq9 > 0.0]
        daily_rows = daily_rows.reset_index()
        daily_rows.drop(columns=["level_1"], inplace=True)

        columns_cal = []
        for col in daily_rows.columns:
            if col not in COLUMNS_INDEX and col not in COLUMNS_HAS:
                columns_cal.append(col)

        if weighted:
            if weighted_days:
                columns_index = COLUMNS_INDEX[1:]
                daily_rows = (
                    daily_rows.set_index(columns_index + COLUMNS_HAS)
                    .groupby(["participant_id"])[columns_cal]
                    .ewm(halflife=ewm_halflife, times=daily_rows["date"])
                    .mean()
                    .reset_index()
                )
            else:
                daily_rows = (
                    daily_rows.set_index(COLUMNS_INDEX + COLUMNS_HAS)
                    .groupby(level=0)
                    .transform(lambda s: s.ewm(span=ewm_span).mean())
                    .reset_index()
                )

        else:  # cumulative previous_num test data without weighting
            if previous_num == 0:
                pass
            else:
                daily_rows = (
                    daily_rows.set_index(COLUMNS_INDEX + COLUMNS_HAS)
                    .groupby(["participant_id"])[columns_cal]
                    .transform(lambda s: s.rolling(previous_num, min_periods=1).mean())
                    .reset_index()
                )
    # add a new column to indicate the number of days between the two studys
    # daily_rows["interval_date"] = (
    #     daily_rows.groupby(["participant_id"])["date"]
    #     .diff()
    #     .dt.days.reset_index()["date"]
    # )
    # daily_rows.fillna(0, inplace=True)
    # daily_rows["date_since_start"] = daily_rows.groupby("participant_id")[
    #     "interval_date"
    # ].cumsum()

    # Now, append the constants to each row
    for const in constants:
        daily_rows = daily_rows.merge(const, on="participant_id")

    return daily_rows, daily_merged
