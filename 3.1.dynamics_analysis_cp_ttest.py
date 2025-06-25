import pandas as pd
from scipy.stats import ttest_ind
import os

output_dir = 'RQ2 Crossing Point Analysis'
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(os.path.join(output_dir, 'crossings_nC.csv'))

def t_test_and_save(task, filename):
    sub = df[df['task'] == task]
    results = []
    for comp_level in sorted(sub['comparison_level'].unique()):
        group = sub[sub['comparison_level'] == comp_level]
        szh_vals = group[group['group'] == 'SZH']['nC'].dropna()
        hc_vals = group[group['group'] == 'HC']['nC'].dropna()
        n_SZH = len(szh_vals)
        n_HC = len(hc_vals)
        if n_HC < 2 or n_SZH < 2:
            continue
        t_stat, p_val = ttest_ind(szh_vals, hc_vals, equal_var=False)
        if p_val < 0.001:
            significance = '***'
        elif p_val < 0.01:
            significance = '**'
        elif p_val < 0.05:
            significance = '*'
        else:
            significance = ''
        results.append({
            'comparison_level': comp_level,
            'SZH_mean': round(szh_vals.mean(), 3),
            'HC_mean': round(hc_vals.mean(), 3),
            't_stat': round(t_stat, 3),
            'p_value': round(p_val, 4),
            'significance': significance,
            'n_SZH': n_SZH,
            'n_HC': n_HC
        })
    result_df = pd.DataFrame(results, columns=[
        'comparison_level', 'SZH_mean', 'HC_mean', 't_stat',
        'p_value', 'significance', 'n_SZH', 'n_HC'
    ])
    result_df.to_csv(os.path.join(output_dir, filename), index=False, float_format="%.4f")

t_test_and_save('spontaneous', 'crossings_nC_ttest_spontaneous.csv')
t_test_and_save('picture', 'crossings_nC_ttest_picture.csv')
