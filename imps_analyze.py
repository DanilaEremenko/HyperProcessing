from pathlib import Path
from typing import List

import pandas as pd

from clf import clf_build
from compare_snapshots import RES_DIR
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from drawing import draw_n_vectors_matplot, draw_n_vectors_plotly
from snapshots_processing import WLS


########################################################################################################################
# --------------------------------- get and draw importance vectors from experiments -----------------------------------
########################################################################################################################
def read_imps(path: str) -> pd.DataFrame:
    with open(path) as fp:
        imp_str = fp.read()
        imp_dict = [
            {'wl': line.split('_')[0], 'imp': float(line.split(':')[1])}
            for line in imp_str.split('\n')
        ]
        return pd.DataFrame(imp_dict).sort_values(['wl'])


CH_STATISTICS = [
    'all_pixels_mean',
    # 'all_pixels_std'
]
IMP_DIR = Path('topic/imp_based_on_mean')
IMPS1 = f'{IMP_DIR}/imp1'
IMPS2 = f'{IMP_DIR}/imp2'


def get_all_imps_df() -> pd.DataFrame:
    imps_df1 = read_imps(IMPS1)
    imps_df2 = read_imps(IMPS2)
    if len(CH_STATISTICS) > 1:
        imps_df1 = imps_df1.groupby('wl').sum().reset_index()
        imps_df2 = imps_df2.groupby('wl').sum().reset_index()

    all_df = pd.merge(left=imps_df1, right=imps_df2, left_on='wl', right_on='wl')

    all_df.rename(columns={'imp_x': 'importance exp 2', 'imp_y': 'importance exp 3'}, inplace=True)

    all_df = all_df.iloc[:-1]

    all_df['wl'] = pd.to_numeric(all_df['wl'])

    return all_df


ALL_DF = get_all_imps_df()


def _draw_imps_plotly():
    fig = go.Figure()

    for key in ['importance exp 2', 'importance exp 3']:
        fig.add_trace(
            go.Scatter(x=ALL_DF['wl'], y=ALL_DF[key], name=key),
        )

    fig.update_layout(
        title='Experiments 2',
        xaxis_title="wavelengths (nm)",
        yaxis_title="importance",
        legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig.show()
    fig.write_html(f'{IMP_DIR}/band_imps.html')


def _draw_imps_matplotlib():
    plt.rcParams["figure.figsize"] = (9, 6)

    for key in ['importance exp 2', 'importance exp 3']:
        plt.plot(ALL_DF['wl'], ALL_DF[key], label=key)

    plt.xlabel("wavelengths (nm)")
    plt.ylabel("importance")
    plt.legend(loc="upper right")
    plt.savefig(f"{IMP_DIR}/band_imps.png", dpi=100)
    plt.clf()


_draw_imps_matplotlib()
_draw_imps_plotly()


########################################################################################################################
# --------------------------------- get wl orders based on different importance sorting --------------------------------
########################################################################################################################
def get_imps_df_by_window():
    window_size = 5
    wl_sort_df = ALL_DF.sort_values('wl', ascending=False)

    windows_imps_list = []
    for start in range(0, len(wl_sort_df) - window_size + 1, 1):
        curr_window = ALL_DF.iloc[start:start + window_size]
        windows_imps_list.append(
            {
                'wls': list(curr_window['wl']),
                'corr': curr_window.corr().iloc[1][2]
            }
        )
    windows_imps_df = pd.DataFrame(windows_imps_list).sort_values('corr', ascending=False)
    return windows_imps_df


def get_imps_df_by_deviation() -> pd.DataFrame:
    df = ALL_DF.copy()
    df['mae'] = (df['importance exp 2'] - df['importance exp 3']).abs()
    return df[['wl', 'mae']].sort_values('mae', ascending=False)


def get_imps_df_by_sum() -> pd.DataFrame:
    df = ALL_DF.copy()
    df['sum'] = df['importance exp 2'] + df['importance exp 3']
    return df[['wl', 'sum']].sort_values('sum', ascending=True)


def get_imps_by_sum_and_closeness():
    df = ALL_DF.copy()
    df['mae'] = (df['importance exp 2'] - df['importance exp 3']).abs()
    df['mae_normalized'] = df['mae'] / df['mae'].max()
    df['sum'] = df['importance exp 2'] + df['importance exp 3']
    df['smart'] = df['sum'] * (1 - df['mae_normalized'])
    return df[['wl', 'smart']].sort_values('smart', ascending=False)


WINDOWS_IMPS = get_imps_df_by_window()
DEVIATION_ASCENDING_IMPS = get_imps_df_by_deviation()
SUM_DESCENDING_IMPS = get_imps_df_by_sum()
SMART_IMPS = get_imps_by_sum_and_closeness()

########################################################################################################################
# --------------------------------- check minimal sufficient subset of channels ----------------------------------------
########################################################################################################################
FEATURES_DF = pd.read_csv(f"{RES_DIR}/features.csv")


def _clf_build(fit_df: pd.DataFrame, eval_df: pd.DataFrame, x_keys: List[str]):
    return clf_build(
        fit_df=fit_df,
        eval_df=eval_df,
        x_keys=x_keys,
        y_key='class_generalized',
        method_name='lr',
        clf_args=dict(max_iter=1e3),
        scaler_fit_on_all=False
    )


def draw_head_importance_dynamic(imp_wls: List[float], x_label_postf: str, save_pref=None):
    f1_dict_exp1 = {'train': [], 'test': [], 'eval': []}
    f1_dict_exp2 = {'train': [], 'test': [], 'eval': []}

    n_important_list = [n_important for n_important in range(1, len(imp_wls), 1)]

    df2 = FEATURES_DF.iloc[0:400]
    df3 = FEATURES_DF.iloc[400:800]
    for n_important in n_important_list:
        print(f"build statistic for {n_important} head important features")

        head_n_features = [f"{wl}_{pred}"
                           for pred in CH_STATISTICS
                           for wl in imp_wls[:n_important]]

        clf_results = _clf_build(fit_df=df2, eval_df=df3, x_keys=head_n_features)
        f1_dict_exp1['train'].append(clf_results['train_f1_phyto'])
        f1_dict_exp1['test'].append(clf_results['test_f1_phyto'])
        f1_dict_exp1['eval'].append(clf_results['eval_f1_phyto'])

        clf_results = _clf_build(fit_df=df3, eval_df=df2, x_keys=head_n_features)
        f1_dict_exp2['train'].append(clf_results['train_f1_phyto'])
        f1_dict_exp2['test'].append(clf_results['test_f1_phyto'])
        f1_dict_exp2['eval'].append(clf_results['eval_f1_phyto'])

    for metrics_dict, split_name in zip([f1_dict_exp1, f1_dict_exp2], ['split 1', 'split 2']):
        args = dict(
            x_list=[n_important_list for i in range(len(metrics_dict))],
            y_list=list(metrics_dict.values()),
            meta_list=[[str(wl) for wl in imp_wls] for i in range(len(f1_dict_exp1))],
            labels=list(metrics_dict.keys()),
            x_label=f'number of head important features {x_label_postf}',
            y_label='f1_score',
            title=split_name,
            save_pref=save_pref if save_pref is None else f"{save_pref} {split_name}"
        )
        draw_n_vectors_matplot(**args)
        draw_n_vectors_plotly(**args)


draw_head_importance_dynamic(imp_wls=list(DEVIATION_ASCENDING_IMPS['wl']),
                             x_label_postf='\n(ordered by mae(train, eval) importance criterion asc ↑)',
                             save_pref=f'{IMP_DIR}/imps_by_importance_mae_asc')

draw_head_importance_dynamic(imp_wls=list(SUM_DESCENDING_IMPS['wl']),
                             x_label_postf='\n(ordered by sum(train, eval) importance criterion desc ↓)',
                             save_pref=f'{IMP_DIR}/imps_by_importance_sum_desc')

draw_head_importance_dynamic(imp_wls=list(SMART_IMPS['wl']),
                             x_label_postf='\n(ordered by sum(train, eval) & closeness(train,eval) importance desc ↓)',
                             save_pref=f'{IMP_DIR}/imps_by_smart_asc')

draw_head_importance_dynamic(imp_wls=list(ALL_DF.sort_values('importance exp 2', ascending=False)['wl']),
                             x_label_postf='\n(ordered by exp 2 importance desc ↓)',
                             save_pref=f'{IMP_DIR}/imps_by_exp_2_desc',
                             )

draw_head_importance_dynamic(imp_wls=list(ALL_DF.sort_values('importance exp 3', ascending=False)['wl']),
                             x_label_postf='\n(ordered by exp 3 importance desc ↓)',
                             save_pref=f'{IMP_DIR}/imps_by_exp_3_desc',
                             )

draw_head_importance_dynamic(imp_wls=WLS,
                             x_label_postf='\n(ordered by wl asc ↑)',
                             save_pref=f'{IMP_DIR}/imps_by_wl_ascending')

draw_head_importance_dynamic(imp_wls=WLS[::-1],
                             x_label_postf='\n(ordered by wl desc ↓)',
                             save_pref=f'{IMP_DIR}/imps_by_wl_descending')
