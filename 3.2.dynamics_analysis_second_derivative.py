import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("sliding_window_analysis.csv")
df['participant'] = df['participant'].astype(str)

df['window_index_norm'] = df.groupby(['participant', 'task'])['window_index'].transform(
    lambda x: 100 * x.rank(method='first', pct=True)
)
np_types = ['TI', 'REF', 'UI', 'FAM', 'ACT', 'InF']

def get_smoothed_and_derivs(x, y, frac):
    smoothed = lowess(y, x, frac=frac, return_sorted=False)
    if len(smoothed) > 5:
        smoothed = savgol_filter(smoothed, 5, 2)
    second_deriv = np.gradient(np.gradient(smoothed))
    return smoothed, second_deriv

records = []
for (pid, task), group in df.groupby(['participant', 'task']):
    group = group.sort_values('window_index_norm')
    x = group['window_index_norm'].values
    for np_type in np_types:
        y = group[np_type].cumsum().values
        frac = 0.2 if 'spontaneous' in task else 0.35  # picture : 0.35, spontaneous : 0.2
        smoothed, second_deriv = get_smoothed_and_derivs(x, y, frac=frac)
        mean_sd_all = np.mean(second_deriv)
        half = int(len(x) / 2)
        mean_sd_first = np.mean(second_deriv[:half])
        mean_sd_second = np.mean(second_deriv[half:])
        records.append({
            'ID': pid,
            'task': task,
            'NP_type': np_type,
            'second_derivative_all': mean_sd_all,
            'second_derivative_first_half': mean_sd_first,
            'second_derivative_second_half': mean_sd_second
        })

result_df = pd.DataFrame(records)


picture_df = result_df[result_df['task'].str.contains('picture')]
picture_value_df = (
    picture_df.groupby('NP_type')[['second_derivative_all', 
                                   'second_derivative_first_half', 
                                   'second_derivative_second_half']]
    .mean()
    .reset_index()
    .rename(columns={
        'second_derivative_all': 'picture_value_all',
        'second_derivative_first_half': 'picture_value_first_half',
        'second_derivative_second_half': 'picture_value_second_half'
    })
)
print(picture_value_df)


group_map = df.groupby(['participant', 'task']).first().reset_index()[['participant', 'task', 'group']]
group_map['participant'] = group_map['participant'].astype(str)
result_df = result_df.merge(group_map, left_on=['ID', 'task'], right_on=['participant', 'task'], how='left')
result_df.drop('participant', axis=1, inplace=True)


main_tasks = ['spontaneous', 'picture']
result_df['main_task'] = result_df['task'].apply(
    lambda x: 'spontaneous' if 'spontaneous' in x else ('picture' if 'picture' in x else x)
)



output_folder = "RQ2 Second Derivative"
os.makedirs(output_folder, exist_ok=True)

metric_cols = [
    ('second_derivative_all', 'all'),
    ('second_derivative_first_half', 'firsthalf'),
    ('second_derivative_second_half', 'secondhalf')
]

for col, colname in metric_cols:
    for task in main_tasks:
        results = []
        for np_type in np_types:
            df_sub = result_df[result_df['main_task'] == task]
            df_sub = df_sub[df_sub['NP_type'] == np_type]
            hc_vals = df_sub[df_sub['group'] == 'HC'][col].dropna()
            szh_vals = df_sub[df_sub['group'] == 'SZH'][col].dropna()
            if len(hc_vals) > 1 and len(szh_vals) > 1:
                tstat, pval = ttest_ind(hc_vals, szh_vals, equal_var=False)
            else:
                tstat, pval = np.nan, np.nan
            results.append({
                'NP_type': np_type,
                't_stat': tstat,
                'p_value': pval,
                'HC_mean': hc_vals.mean(),
                'SZH_mean': szh_vals.mean()
            })
        results_df = pd.DataFrame(results)

        if len(results_df) > 1:
            reject, pvals_bonf, _, _ = multipletests(results_df['p_value'], method='bonferroni')
            results_df['p_bonferroni'] = pvals_bonf
            results_df['bonferroni_significant'] = reject
        else:
            results_df['p_bonferroni'] = results_df['p_value']
            results_df['bonferroni_significant'] = False
        out_fn = f"f2_{colname}_ttest_{task}_bonferroni.csv"
        out_path = os.path.join(output_folder, out_fn)
        results_df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"Saved: {out_path}")


subdf = df[(df['participant'] == '1') & (df['task'].str.contains('picture'))].sort_values('window_index_norm')
x = subdf['window_index_norm']
np_type = 'TI'
y = subdf[np_type].cumsum()
smoothed, _ = get_smoothed_and_derivs(x, y, frac=0.35)
plt.plot(x, y, label="Raw")
plt.plot(x, smoothed, label="LOWESS smoothed", linewidth=2)
plt.xlabel("Window index (%)")
plt.ylabel("Cumulative count")
plt.title("Example: Smoothed Curve")
plt.legend()
plt.tight_layout()
plt.show()

pic_df = df[df['task'].str.contains('picture')].copy()
if 'group' not in pic_df.columns:
    group_map = df.groupby(['participant', 'task']).first().reset_index()[['participant', 'task', 'group']]
    pic_df = pic_df.merge(group_map, on=['participant', 'task'], how='left')

window_bins = np.linspace(0, 100, 51)
bin_centers = window_bins[:-1] + (window_bins[1]-window_bins[0])/2

results = []
for (pid, task), group in pic_df.groupby(['participant', 'task']):
    x = group['window_index_norm'].values
    y = group['InF'].cumsum().values
    if len(x) < 6:
        continue
    smoothed = lowess(y, x, frac=0.35, return_sorted=False)
    smoothed = savgol_filter(smoothed, 5, 2)
    second_deriv = np.gradient(np.gradient(smoothed))
    bins = pd.cut(x, bins=window_bins, labels=False, include_lowest=True)
    for b in np.unique(bins[~pd.isnull(bins)]):
        mask = bins == b
        if mask.sum() == 0:
            continue
        results.append({
            'participant': pid,
            'group': group['group'].iloc[0],
            'window_bin': b,
            'second_derivative': np.mean(second_deriv[mask])
        })
