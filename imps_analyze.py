import pandas as pd


def read_imps(path: str) -> pd.DataFrame:
    with open(path) as fp:
        imp_str = fp.read()
        imp_dict = [
            {'wl': line.split('_')[0], 'imp': float(line.split(':')[1])}
            for line in imp_str.split('\n')
        ]
        return pd.DataFrame(imp_dict).sort_values(['wl'])


def get_all_imps_df() -> pd.DataFrame:
    imps_df1 = read_imps('sub-wheat_comparison_with_indexes_filtered_each_wl_imp_analyze_cropped/imp1')
    imps_df2 = read_imps('sub-wheat_comparison_with_indexes_filtered_each_wl_imp_analyze_cropped/imp2')

    all_df = pd.merge(left=imps_df1, right=imps_df2, left_on='wl', right_on='wl')

    all_df.rename(columns={'imp_x': 'importance exp 2', 'imp_y': 'importance exp 3'}, inplace=True)

    all_df = all_df.iloc[:-1]

    all_df['wl'] = pd.to_numeric(all_df['wl'])

    return all_df


ALL_DF = get_all_imps_df()


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


def get_imps_df_by_residuals():
    df = ALL_DF.copy()
    df['mae'] = (df['importance exp 2'] - df['importance exp 3']).abs()
    return df[['wl', 'mae']].sort_values('mae', ascending=False)


WINDOWS_IMPS = get_imps_df_by_window()
RESIDUALS_IMPS = get_imps_df_by_residuals()


def _draw_imps_plotly():
    import plotly.graph_objs as go

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
    fig.write_html('band_imps.html')


def _draw_imps_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (9, 6)

    for key in ['importance exp 2', 'importance exp 3']:
        plt.plot(ALL_DF['wl'], ALL_DF[key], label=key)

    plt.legend(loc="upper right")
    plt.savefig("band_imps.png", dpi=100)
    plt.clf()

# _draw_imps_matplotlib()
# _draw_imps_plotly()
