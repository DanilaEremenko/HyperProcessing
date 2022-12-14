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
IMP_DIR = Path('topic/svc_imp_based_on_mean')
IMPS1 = f'{IMP_DIR}/imp1'
IMPS2 = f'{IMP_DIR}/imp2'


def get_all_imps_df() -> pd.DataFrame:
    imps_df1 = read_imps(IMPS1)
    imps_df2 = read_imps(IMPS2)
    if len(CH_STATISTICS) > 1:
        imps_df1 = imps_df1.groupby('wl').sum().reset_index()
        imps_df2 = imps_df2.groupby('wl').sum().reset_index()

    all_df = pd.merge(left=imps_df1, right=imps_df2, left_on='wl', right_on='wl')

    all_df.rename(columns={'imp_x': 'importance exp 1', 'imp_y': 'importance exp 2'}, inplace=True)

    all_df = all_df.iloc[:-1]

    all_df['wl'] = pd.to_numeric(all_df['wl'])

    return all_df


ALL_DF = get_all_imps_df()


def _draw_imps_plotly():
    fig = go.Figure()

    for key in ['importance exp 1', 'importance exp 2']:
        fig.add_trace(
            go.Scatter(x=ALL_DF['wl'], y=ALL_DF[key], name=key),
        )

    fig.update_layout(
        title='Experiments 2',
        xaxis_title="wavelengths (nm)",
        yaxis_title="importance",
        legend_title="Legend Title",
        font=dict(
            family="Times New Roman",
            size=14,
            color="RebeccaPurple"
        )
    )
    fig.show()
    fig.write_html(f'{IMP_DIR}/band_imps.html')


def _draw_imps_matplotlib():
    plt.rcParams["figure.figsize"] = (9, 6)

    for key in ['importance exp 1', 'importance exp 2']:
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
    df['mae'] = (df['importance exp 1'] - df['importance exp 2']).abs()
    return df[['wl', 'mae']].sort_values('mae', ascending=False)


def get_imps_df_by_sum() -> pd.DataFrame:
    df = ALL_DF.copy()
    df['sum'] = df['importance exp 1'] + df['importance exp 2']
    return df[['wl', 'sum']].sort_values('sum', ascending=True)


def get_imps_by_sum_and_closeness():
    df = ALL_DF.copy()
    df['mae'] = (df['importance exp 1'] - df['importance exp 2']).abs()
    df['mae_normalized'] = df['mae'] / df['mae'].max()
    df['sum'] = df['importance exp 1'] + df['importance exp 2']
    df['smart'] = df['sum'] * (1 - df['mae_normalized'])
    return df[['wl', 'smart']].sort_values('smart', ascending=False)


WINDOWS_IMPS = get_imps_df_by_window()
DEVIATION_ASCENDING_IMPS = get_imps_df_by_deviation()
SUM_DESCENDING_IMPS = get_imps_df_by_sum()
SMART_IMPS = get_imps_by_sum_and_closeness()

draw_n_vectors_matplot(
    x_list=[SMART_IMPS.sort_values('wl')['wl']],
    y_list=[SMART_IMPS.sort_values('wl')['smart']],
    labels=['joined importance'],
    meta_list=[],
    x_label='wavelengths (nm)',
    y_label='joined importance',
    title='',
    save_pref=f"{IMP_DIR}/band_imps_smart"
)

########################################################################################################################
# --------------------------------- check minimal sufficient subset of channels ----------------------------------------
########################################################################################################################
FEATURES_DF = pd.read_csv(f"{RES_DIR}/features.csv")


def _clf_build(fit_df: pd.DataFrame, eval_df: pd.DataFrame, x_keys: List[str]):
    method_name = IMP_DIR.name.split('_')[0]
    return clf_build(
        fit_df=fit_df,
        eval_df=eval_df,
        x_keys=x_keys,
        y_key='class_generalized',
        clf_args=dict(C=1, kernel='linear') if method_name == 'svc' else {},
        method_name=method_name,
        scaler_fit_on_all=False
    )


def draw_head_importance_dynamic(imp_wls: List[float], x_label_postf: str, save_pref=None):
    kappa_dict_exp1 = {'train': [], 'test': [], 'eval': []}
    kappa_dict_exp2 = {'train': [], 'test': [], 'eval': []}

    n_important_list = [n_important for n_important in range(1, len(imp_wls), 1)]

    # df2 = FEATURES_DF.iloc[0:400]
    # df3 = FEATURES_DF.iloc[400:800]
    df2 = pd.concat([FEATURES_DF.iloc[0:200], FEATURES_DF.iloc[400:600]])
    df3 = pd.concat([FEATURES_DF.iloc[600:800], FEATURES_DF.iloc[900:1100]])
    for n_important in n_important_list:
        print(f"build statistic for {n_important} head important features")

        head_n_features = [f"{wl}_{pred}"
                           for pred in CH_STATISTICS
                           for wl in imp_wls[:n_important]]

        clf_results = _clf_build(fit_df=df2, eval_df=df3, x_keys=head_n_features)
        kappa_dict_exp1['train'].append(clf_results['train_kappa'])
        kappa_dict_exp1['test'].append(clf_results['test_kappa'])
        kappa_dict_exp1['eval'].append(clf_results['eval_kappa'])

        clf_results = _clf_build(fit_df=df3, eval_df=df2, x_keys=head_n_features)
        kappa_dict_exp2['train'].append(clf_results['train_kappa'])
        kappa_dict_exp2['test'].append(clf_results['test_kappa'])
        kappa_dict_exp2['eval'].append(clf_results['eval_kappa'])

    for metrics_dict, split_name in zip([kappa_dict_exp1, kappa_dict_exp2], ['split 1', 'split 2']):
        args = dict(
            x_list=[n_important_list for i in range(len(metrics_dict))],
            y_list=list(metrics_dict.values()),
            meta_list=[[str(wl) for wl in imp_wls] for i in range(len(kappa_dict_exp1))],
            labels=list(metrics_dict.keys()),
            x_label=f'number of head important features {x_label_postf}',
            y_label='kappa index',
            title=split_name,
            save_pref=save_pref if save_pref is None else f"{save_pref} {split_name}"
        )
        draw_n_vectors_matplot(**args)
        draw_n_vectors_plotly(**args)




draw_head_importance_dynamic(imp_wls=list(DEVIATION_ASCENDING_IMPS['wl']),
                             x_label_postf='\n(ordered by mae(train, eval) importance criterion asc ↑)',
                             save_pref=f'{IMP_DIR}/convergence_by_importance_mae_asc')

draw_head_importance_dynamic(imp_wls=list(SUM_DESCENDING_IMPS['wl']),
                             x_label_postf='\n(ordered by sum(train, eval) importance criterion desc ↓)',
                             save_pref=f'{IMP_DIR}/convergence_by_importance_sum_desc')

draw_head_importance_dynamic(imp_wls=list(SMART_IMPS['wl']),
                             x_label_postf='\n(ordered by joined importance desc ↓)',
                             save_pref=f'{IMP_DIR}/convergence_by_smart_asc')

draw_head_importance_dynamic(imp_wls=list(ALL_DF.sort_values('importance exp 2', ascending=False)['wl']),
                             x_label_postf='\n(ordered by exp 2 importance desc ↓)',
                             save_pref=f'{IMP_DIR}/convergence_by_exp_2_desc',
                             )

draw_head_importance_dynamic(imp_wls=list(ALL_DF.sort_values('importance exp 3', ascending=False)['wl']),
                             x_label_postf='\n(ordered by exp 3 importance desc ↓)',
                             save_pref=f'{IMP_DIR}/convergence_by_exp_3_desc',
                             )

draw_head_importance_dynamic(imp_wls=WLS,
                             x_label_postf='\n(ordered by wl asc ↑)',
                             save_pref=f'{IMP_DIR}/convergence_by_wl_ascending')

draw_head_importance_dynamic(imp_wls=WLS[::-1],
                             x_label_postf='\n(ordered by wl desc ↓)',
                             save_pref=f'{IMP_DIR}/convergence_by_wl_descending')
