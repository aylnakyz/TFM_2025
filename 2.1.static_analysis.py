import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import os

df = pd.read_csv('dataframe_static.csv')

np_type_order = ['TI', 'REF', 'UI', 'FAM', 'ACT', 'InF']
df['Np Type'] = pd.Categorical(df['Np Type'], categories=np_type_order, ordered=True)

def add_picture_average_rows(df):
    pict_mask = df['Task'].isin(['picture1', 'picture2', 'picture3'])
    df_pict = df[pict_mask].copy()
    pict_avg = df_pict.groupby(['Par ID', 'Group', 'Np Type']).agg(
        num_words=('num_words', 'mean'),
        nt=('nt', 'mean'),
        nt_nword=('nt/nword', 'mean'),
        nt_totalnt=('nt/totalnt', 'mean')
    ).reset_index()
    pict_avg['Task'] = 'picture'
    pict_avg['Group_Task'] = pict_avg['Group'] + '_picture'
    pict_avg['Np Type'] = pd.Categorical(pict_avg['Np Type'], categories=np_type_order, ordered=True)
    pict_avg = pict_avg.rename(columns={'nt_nword': 'nt/nword', 'nt_totalnt': 'nt/totalnt'})
    return pd.concat([df, pict_avg], ignore_index=True)

df = add_picture_average_rows(df)
df['Group_Task'] = df['Group'] + '_' + df['Task']

output_dir = "RQ1 Static Analysis"
os.makedirs(output_dir, exist_ok=True)

tasks = ['spontaneous', 'picture']
metrics = ['nt/totalnt', 'nt/nword']

def significance(p):
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

for task in tasks:
    for metric in metrics:
        results = []
        for nptype in df['Np Type'].cat.categories:
            sub_df = df[(df['Task'] == task) & (df['Np Type'] == nptype)]
            group1 = sub_df[sub_df['Group'] == 'SZH'][metric].dropna()
            group2 = sub_df[sub_df['Group'] == 'HC'][metric].dropna()
            if len(group1) > 1 and len(group2) > 1:
                tstat, pval = ttest_ind(group1, group2, equal_var=False)
            else:
                tstat, pval = np.nan, np.nan

            SZH_mean = round(group1.mean(), 5) if not np.isnan(group1.mean()) else np.nan
            HC_mean = round(group2.mean(), 5) if not np.isnan(group2.mean()) else np.nan
            tstat_rounded = round(tstat, 3) if not np.isnan(tstat) else np.nan
            pval_rounded = round(pval, 4) if not np.isnan(pval) else np.nan

            results.append({
                'Task': task,
                'Np Type': nptype,
                'Metric': metric,
                'SZH_mean': SZH_mean,
                'HC_mean': HC_mean,
                't-stat': tstat_rounded,
                'p-value': pval_rounded,
                'Significance': significance(pval)
            })

        results_df = pd.DataFrame(results)

        n_tests = results_df.shape[0]
        results_df['p_bonf'] = results_df['p-value'] * n_tests
        results_df['p_bonf'] = results_df['p_bonf'].clip(upper=1)
        results_df['Significance_bonf'] = results_df['p_bonf'].apply(significance)

        task_str = "Spontaneous" if task == "spontaneous" else "Picture"
        metric_str = "nt_totalnt" if metric == "nt/totalnt" else "nt_nword"
        filename = f"{task_str}_{metric_str}_ttest_Bonferroni.csv"
        filepath = os.path.join(output_dir, filename)

        results_df.to_csv(filepath, index=False, float_format="%.4f")

print("Analysis completed.")
