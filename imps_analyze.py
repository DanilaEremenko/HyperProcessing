import pandas as pd


def read_imps(path: str) -> pd.DataFrame:
    with open(path) as fp:
        imp_str = fp.read()
        imp_dict = [
            {'wl': line.split('_')[0], 'imp': float(line.split(':')[1])}
            for line in imp_str.split('\n')
        ]
        return pd.DataFrame(imp_dict).sort_values(['wl'])


imps_df1 = read_imps('sub-wheat_comparison_with_indexes_filtered_each_wl_imp_analyze_cropped/imp1')
imps_df2 = read_imps('sub-wheat_comparison_with_indexes_filtered_each_wl_imp_analyze_cropped/imp2')

all_df = pd.merge(left=imps_df1, right=imps_df2, left_on='wl', right_on='wl')

all_df.rename(columns={'imp_x': 'importance exp 2', 'imp_y': 'importance exp 3'}, inplace=True)

all_df = all_df.iloc[:-1]

all_df['wl'] = pd.to_numeric(all_df['wl'])

import plotly.graph_objs as go

fig = go.Figure()

for key in ['importance exp 2', 'importance exp 3']:
    fig.add_trace(
        go.Scatter(x=all_df['wl'], y=all_df[key], name=key),
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

# fig.show()
fig.write_html('band_imps.html')
