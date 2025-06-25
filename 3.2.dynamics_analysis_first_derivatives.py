import os
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import ttest_ind

INPUT_FILE = "sliding_window_analysis.csv"
OUT_DIR = "RQ2 First Derivative"
os.makedirs(OUT_DIR, exist_ok=True)
np_types = ['TI', 'REF', 'UI', 'FAM', 'ACT', 'InF']

df = pd.read_csv(INPUT_FILE)

df_long = df.melt(
    id_vars=['window_index', 'group', 'participant', 'task'],
    value_vars=np_types,
    var_name='np_type',
    value_name='count'
)

group_cols = ['participant', 'task', 'np_type']

def add_features_with_group(group):
    group = group.sort_values('window_index').reset_index(drop=True)
    n = len(group)
    group['window_percentage'] = np.linspace(0, 100, n) if n > 1 else 0
    group['cumulative'] = group['count'].cumsum()
    return group

df_long = (
    df_long
    .groupby(group_cols, as_index=False)
    .apply(add_features_with_group)
    .reset_index(drop=True)
)

valid_combinations = (
    df_long.groupby(['participant', 'task', 'np_type'])['count']
    .sum().reset_index()
)
valid_combinations = valid_combinations[valid_combinations['count'] > 0]
valid_set = set(zip(valid_combinations['participant'], valid_combinations['task'], valid_combinations['np_type']))


def lowess_smooth(x, y, frac):
    if len(x) < 3:
        return y
    y_smooth = lowess(y, x, frac=frac, return_sorted=False)
    return y_smooth

def calculate_derivatives(x, y_smooth):
    return np.gradient(y_smooth, x)

results = []
tasks = ['spontaneous']

for pid, p_df in df_long.groupby('participant'):
    for task in tasks:
        for np_type in np_types:
            if (pid, task, np_type) not in valid_set:
                continue
            t_df = p_df[p_df['task'] == task]
            n_df = t_df[t_df['np_type'] == np_type]
            n_df = n_df.sort_values('window_percentage')
            x = n_df['window_percentage'].values
            y = n_df['cumulative'].values
            frac = 0.07
            y_smooth = lowess_smooth(x, y, frac)
            derivatives = calculate_derivatives(x, y_smooth)
            mean_derivative = np.mean(derivatives) if len(derivatives) > 0 else np.nan
            results.append({
                'participant': pid,
                'task': task,
                'np_type': np_type,
                "curve_derivative_mean": mean_derivative
            })

    for np_type in np_types:
        picture_valid = any((pid, pictask, np_type) in valid_set for pictask in ['picture1', 'picture2', 'picture3'])
        if not picture_valid:
            continue
        dfs = []
        for pictask in ['picture1', 'picture2', 'picture3']:
            if (pid, pictask, np_type) not in valid_set:
                continue
            n_df = p_df[(p_df['task'] == pictask) & (p_df['np_type'] == np_type)]
            n_df = n_df.sort_values('window_percentage')
            dfs.append(n_df[['window_percentage', 'cumulative']].set_index('window_percentage'))
        if len(dfs) == 0:
            continue
        merged = pd.concat(dfs, axis=1).fillna(0)
        mean_y = merged.mean(axis=1).values
        x = merged.index.values
        frac = 0.5 if len(x) < 30 else 0.07
        y_smooth = lowess_smooth(x, mean_y, frac)
        derivatives = calculate_derivatives(x, y_smooth)
        mean_derivative = np.mean(derivatives) if len(derivatives) > 0 else np.nan
        results.append({
            'participant': pid,
            'task': 'picture',
            'np_type': np_type,
            "curve_derivative_mean": mean_derivative
        })


results_df = pd.DataFrame(results)

participant_to_group = df[['participant', 'group']].drop_duplicates().set_index('participant')['group']
results_df['group'] = results_df['participant'].map(participant_to_group)

results_df = results_df.drop_duplicates(subset=['participant', 'task', 'np_type'])

results_df = results_df[results_df['task'].isin(['spontaneous', 'picture'])].copy()

results_df.to_csv(f"{OUT_DIR}/curve_derivative_mean_by_participant.csv", index=False, float_format="%.6f")


def ttest_and_bonferroni(task, df, out_dir):
    out_rows = []
    for np_type in np_types:
        szh_data = df[(df['group'] == 'SZH') & (df['task'] == task) & (df['np_type'] == np_type)]['curve_derivative_mean'].dropna()
        hc_data  = df[(df['group'] == 'HC')  & (df['task'] == task) & (df['np_type'] == np_type)]['curve_derivative_mean'].dropna()
        if (len(szh_data) >= 2) and (len(hc_data) >= 2):
            t_stat, p_val = ttest_ind(szh_data, hc_data, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan

        if pd.isnull(p_val):
            sig = ''
        elif p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = ''
        out_rows.append({
            'task': task,
            'np_type': np_type,
            'group1': 'SZH',
            'group2': 'HC',
            'mean_szh': szh_data.mean(),
            'mean_hc': hc_data.mean(),
            't_stat': t_stat,
            'p_val': p_val,
            'n_szh': len(szh_data),
            'n_hc': len(hc_data),
            'significance': sig
        })
    ttest_df = pd.DataFrame(out_rows)

    n_tests = (ttest_df['p_val'].notnull()).sum()
    ttest_df['p_bonferroni'] = (ttest_df['p_val'] * n_tests).clip(upper=1.0)

    def signif_bonf(p):
        if pd.isnull(p):
            return ''
        elif p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    ttest_df['significant_bonferroni'] = ttest_df['p_bonferroni'].apply(signif_bonf)

    ttest_df.to_csv(f"{out_dir}/ttest_curve_derivative_mean_{task}.csv", index=False, float_format="%.6f")
    return ttest_df


_ = ttest_and_bonferroni('spontaneous', results_df, OUT_DIR)
_ = ttest_and_bonferroni('picture', results_df, OUT_DIR)

print("Analysis completed")

