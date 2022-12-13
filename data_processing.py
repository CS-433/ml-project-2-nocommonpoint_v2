from collections import defaultdict

import pandas as pd
import numpy as np

# regression|classification & value|diff
def load_phq9_targets(phq9_path, type='regression', target='value', derived_sum=True):
    csv = pd.read_csv(phq9_path)

    # TODO: derived sum is used in the csv right now

#     if derived_sum:
#         s = csv.loc[:, 'phq9_1':'phq9_9'].sum(axis=1)
#     else:
#         s = csv['phq9_sum']

    target_diff = False
    if target == 'diff':
        target_diff = True

    if type == 'regression':
        target_col = 'phq9_sum_diff' if target_diff else 'phq9_sum'
    elif type == 'classification':
        target_col = 'phq9_level_diff' if target_diff else 'phq9_level'
        csv[target_col] -= 1
    else:
        raise ValueError(f'Invalid PHQ9 target type {type}')

    csv.rename(columns={'phq9Date': 'date', target_col: 'target'}, inplace=True)

    csv.drop(csv.columns.difference(['participant_id', 'date', 'target']), axis=1, inplace=True)

    if target_diff:
        csv.dropna(axis=0, inplace=True) # remove NaN rows, useful for target_diff

    return csv

def load_locations(loc_path):
    # Nothing superfluous. Nice!
    return pd.read_csv(loc_path)

def load_demographics(dem_path):
    csv = pd.read_csv(dem_path)

    # Remove some extra columns
    return csv.drop(['Unnamed: 0', 'startdate', 'study'], axis=1)

def load_passive_mobility(pm_path):
    csv = pd.read_csv(pm_path)

    csv.drop(['week'], axis=1, inplace=True)

    csv.came_to_work = csv.came_to_work.astype('float')
    
    csv.rename(columns={'dt_passive': 'date'}, inplace=True)

    return csv
    
def load_passive_phone(pf_path):
    csv = pd.read_csv(pf_path)

    return csv.drop(['week'], axis=1)

def combine(phq9: pd.DataFrame, dailies=None, constants=None, prev_phq9=False, daily_reduction=['mean', 'std']):
    """
    dailies: (name, csv) sequence, csv's are expected to have participant_id and date
    constants: (name, csv) sequence too, csv's are expected to have participant_id only
    """

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

    # Remember that PHQ9 is also daily.
    # Dailies should be (name, csv) pairs.

    # Also, let's add an extra column has_* to check for daily attributes later
    has_keys = ['has_phq9']
    phq9['has_phq9'] = True

    for name, csv in dailies:
        k = f'has_{name}'
        csv[k] = True
        has_keys.append(k)

    # Now, merge everything. Use an 'outer' merge so that all entries are included,
    # and those who don't exist for that date get NaNs
    daily_merged = phq9
    for name, csv in dailies:
        daily_merged = daily_merged.merge(csv, on=['participant_id', 'date'], how='outer')
    daily_merged.sort_values(by=['participant_id', 'date'], inplace=True)

    # To make things look a bit cleaner, replace NaN with False for has_* columns
    for has_key in has_keys:
        daily_merged[has_key].fillna(False, inplace=True)

    if prev_phq9:
        shifted_target = phq9.groupby('participant_id').apply(lambda x: x['target'].shift(1))
        shifted_target.fillna(0, inplace=True)
        daily_merged['prev_target'] = shifted_target.reset_index(drop=True)

    # Now we need to do the reduction. Now, I'm sure there is some sort of

    # use a closure for grouping so we can cache some variables 
    
    # note how to aggregate each column
    # we keep the last value for target and date (which is the phq9 date)
    # use the max for has_* keys so that it is 1 if at least one row is 1
    # the others are features, and we aggregate them with mean and std (daily_reduction)

    def has_filt(col):
        return col.startswith('has_')
    
    def last_filt(col):
        return col in ('target', 'date', 'prev_target')

    def feature_filt(col):
        return col != 'participant_id' and not (has_filt(col) or last_filt(col))
   

    cols = daily_merged.columns
    aggdict = {
        **{c: 'last' for c in cols if last_filt(c)},
        **{c: 'max' for c in cols if has_filt(c)},
        **{c: daily_reduction for c in cols if feature_filt(c)}
    }

    # now, build a rename dict to quickly rename columns
    rename_dict = {}
    for c in cols:
        if last_filt(c):
            rename_dict[c, 'last'] = c
        elif has_filt(c):
            rename_dict[c, 'max'] = c
        else:
            for reduct in daily_reduction:
                rename_dict[c, reduct] = c + '_' + reduct

    # finally, the grouping closure
    def _group_and_reduce_phq9(df):
        inds = _group_by_phq9(df)
        aggframe = df.groupby(inds).agg(aggdict)
        aggframe.columns = [rename_dict[c] for c in aggframe.columns.to_flat_index()]
        return aggframe

    pgrp = daily_merged.groupby('participant_id')
    daily_rows = pgrp.apply(_group_and_reduce_phq9)

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
        daily_rows = daily_rows.merge(const, on='participant_id')

    # Add previous phq9 as context
#     if prev_phq9:
#         p = phq9.groupby('participant_id').apply(
#             lambda x: pd.concat(
#                 (x, x.rename({'target': 'prev_target'}, axis=1)['prev_target'].shift(1)), axis=1))

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

def rf_preprocess(df):
    target_columns = [c for c in df.columns if c.startswith('has_')]
    return df.drop(target_columns, axis=1)

def xy_split(csv):
    y = csv['target'].to_numpy()
    x = csv.drop(csv.columns.intersection(['target', 'participant_id', 'date']), axis=1).to_numpy()
    return x, y

def train_test_split_participant(csv, test_size=0.1, test_take_first=0, random_state=None):
    participants = csv.participant_id.unique()
    np.random.seed(random_state)
    np.random.shuffle(participants)
    cutoff = int(np.ceil(participants.size * test_size))
    
    train_csv = csv[csv.participant_id.isin(participants[cutoff:])]
    test_csv = csv[csv.participant_id.isin(participants[:cutoff])]

    if test_take_first > 0:
#         if 'date' in test_csv.columns:
#             test_csv.sort_values(by=['participant_id', 'date'], inplace=True)

        train_part = test_csv.groupby('participant_id').head(test_take_first)
        test_part = test_csv.groupby('participant_id').tail(-test_take_first)

        final_test_csv = test_part
        final_train_csv = pd.concat((train_csv, train_part))

        train_count = final_train_csv.participant_id.unique().size
        test_count = final_test_csv.participant_id.unique().size
        train_pct = 100 * train_count / (train_count + test_count)
        test_pct = 100 * test_count / (train_count + test_count)

        print(f'After test_take_first={test_take_first}, have {train_pct:.2f}% ({train_count}) '
              f'vs. {test_pct:.2f} ({test_count}) participants in train/test set')
    else:
        final_test_csv, final_train_csv = (test_csv, train_csv)

    x_train, y_train = xy_split(final_train_csv)
    x_test, y_test = xy_split(final_test_csv)

    return x_train, x_test, y_train, y_test

# def _reduce_rows(rows, method='mean'):
#     # rows are sorted by date (and do not contain participant ids)
#     # now, we just combine everything between has_phq9 rows, somehow
#     reduced = []
#     for row in rows:
        
