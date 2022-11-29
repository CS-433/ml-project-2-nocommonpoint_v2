import pandas as pd
import numpy as np

# regression|classification & value|diff
def load_phq9_targets(phq9_path, type='regression', target='value', sum_phq9 = False, phq9_index = 0, derived_sum=True,):
    csv = pd.read_csv(phq9_path)

    # TODO: derived sum is used in the csv right now

#     if derived_sum:
#         s = csv.loc[:, 'phq9_1':'phq9_9'].sum(axis=1)
#     else:
#         s = csv['phq9_sum']
    phq9_targets_value = value_phq9_target(sum_phq9, phq9_index)
    target_diff = False
    if target == 'diff':
        target_diff = True

    if type == 'regression':
        target_col = 'phq9_sum_diff' if target_diff else phq9_targets_value
    elif type == 'classification':
        target_col = 'phq9_level_diff' if target_diff else 'phq9_level'
    else:
        raise ValueError(f'Invalid PHQ9 target type {type}')

    csv.rename(columns={'phq9Date': 'date', target_col: 'target'}, inplace=True)

    csv.drop(csv.columns.difference(['participant_id', 'date', 'target']), axis=1, inplace=True)

    if target_diff:
        csv.dropna(axis=0, inplace=True) # remove NaN rows, useful for target_diff

    return csv


def value_phq9_target(sum_phq9, phq9_index):
    #the target is value
    #if sum_phq9 is true, use the sum provided by dataset
    #if phq9_index is not 0, the target is phq9_index (eg:phq9_3)
    #if phq_index is 0, target is sum of phq9 scores
    if (phq9_index > 0) & (phq9_index < 10):
        value_target = 'phq9_' + str(phq9_index)
    elif phq9_index == 0:
        value_target = 'sum_phq9' if sum_phq9 else 'phq9_sum'
        print(value_target)
    else: raise ValueError(f'Invalid PHQ9 index {phq9_index}')
    return value_target





def load_locations(loc_path):
    # Nothing superfluous. Nice!
    return pd.read_csv(loc_path)

def load_demographics(dem_path):
    csv = pd.read_csv(dem_path)

    # Remove some extra columns
    return csv.drop(['Unnamed: 0', 'startdate', 'study'], axis=1)

def combine(phq9: pd.DataFrame, dailies=None, constants=None, daily_reduction='mean'):
    """
    dailies: (name, csv) sequence, csv's are expected to have participant_id and date
    constants: (name, csv) sequence too, csv's are expected to have participant_id only
    """

    # How does this work?
    # Basically, "combine" daily data between multiple phq9 points.
    # example:
    # phq9  daily0  daily1
    # 3.00  718     333
    # NaN   333     444
    # NaN   555     666
    # 4.00  NaN     NaN 
    # becomes (with mean reduction):
    # phq9  daily0  daily1
    # 3.00  718     333
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

    # Now we need to do the reduction. Now, I'm sure there is some sort of
    # .groupby() magic that can handle this, but I'll do it the lazy way for now.
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

    return daily_rows
  
def _group_and_reduce_phq9(df, method='mean'):
    inds = _group_by_phq9(df)

    return df.groupby(inds).agg('mean')

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

def rf_preprocess(df):
    target_columns = [c for c in df.columns if c.startswith('has_')]
    return df.drop(target_columns, axis=1)

def xy_split(csv):
    y = csv['target'].to_numpy()
    x = csv.drop(csv.columns.intersection(['target', 'participant_id', 'date']), axis=1).to_numpy()
    return x, y

# def _reduce_rows(rows, method='mean'):
#     # rows are sorted by date (and do not contain participant ids)
#     # now, we just combine everything between has_phq9 rows, somehow
#     reduced = []
#     for row in rows:


def load_baselinephq9(basephq9_path, type='regression'):
    csv = pd.read_csv(basephq9_path)

    # Remove some extra columns
    return csv.drop(['Unnamed: 0', 'base_baselinePHQ9date', 'study'], axis=1)
        
def load_phq2_targets(phq2_path, type='regression', target='value', phq2_index = 0):
    #if phq2_index = 0, target is sum of phq2 questions, otherwise target is each phq2 questions
    csv = pd.read_csv(phq2_path)
    if (phq2_index == 1) |(phq2_index == 2) :
        phq2_targets_value = 'phq2_' + str(phq2_index)
    elif phq2_index == 0:
        phq2_targets_value = 'phq2_sum'
    else:
        raise ValueError(f'Invalid PHQ2 index {phq2_index}')

    target_diff = False
    if target == 'diff':
        target_diff = True

    if type == 'regression':
        target_col = 'phq2_sum_diff' if target_diff else phq2_targets_value
    else:
        raise ValueError(f'Invalid PHQ9 target type {type}')

    csv.rename(columns={target_col: 'target'}, inplace=True)

    csv.drop(csv.columns.difference(['participant_id', 'date', 'target']), axis=1, inplace=True)

    if target_diff:
        csv.dropna(axis=0, inplace=True) # remove NaN rows, useful for target_diff
    return csv

