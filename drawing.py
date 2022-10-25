from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.manifold import TSNE

from clf import clf_build
from snapshots_processing import SnapshotMeta, BANDS_DICT
import matplotlib.pyplot as plt


def draw_snapshots_as_reflectance(classes_dict: Dict[str, List[SnapshotMeta]], res_path: str,
                                  x_range: Optional[tuple] = None, y_range: Optional[tuple] = None, mode='ch'):
    for coord_range in (x_range, y_range): assert len(coord_range) == 2

    max_snapshots_in_class = max([len(snapshots) for snapshots in classes_dict.values()])
    fig = make_subplots(
        rows=max_snapshots_in_class,
        cols=len(classes_dict),
        subplot_titles=[snapshot.name for class_snapshots in classes_dict.values() for snapshot in class_snapshots]
    )
    colors = [('0', '180', '0'), ('255', '153', '153'), ('250', '0', '0'), ('113', '12', '12'), ('10', '10', '10')]

    assert len(colors) >= len(classes_dict)

    for col_i, ((group_key, group_list), color_tup) in enumerate(zip(classes_dict.items(), colors)):
        for row_i, snapshot_meta in enumerate(group_list):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta
            band_value = snapshot_meta.bands['all']
            if mode == 'ch':
                # vector of wavelengths representation
                fig.add_trace(
                    go.Scatter(
                        name=group_key,
                        legendgroup=group_key,
                        showlegend=row_i == 0,
                        x=band_value.wave_lengths,
                        y=band_value.mean_in_ch_by_px,
                        line=dict(color=f"rgba({','.join(color_tup)}, 100)")
                    ),
                    row=row_i + 1, col=col_i + 1
                )
                fig.add_trace(
                    go.Scatter(
                        name=group_key,
                        legendgroup=group_key,
                        showlegend=row_i == 0,
                        # x, then x reversed
                        x=[*band_value.wave_lengths, *band_value.wave_lengths[::-1]],
                        mode="markers+lines",
                        fill='toself',
                        line=dict(color=f"rgba({','.join(color_tup)}, 0)"),
                        # upper, then lower reversed
                        y=[*band_value.left_dev_in_chs_by_px,
                           *band_value.right_dev_in_chs_by_px[::-1]]
                    ),
                    row=row_i + 1, col=col_i + 1
                )
            elif mode == 'px':
                fig.add_trace(
                    go.Scatter(
                        name=group_key,
                        legendgroup=group_key,
                        showlegend=row_i == 0,
                        x=list(range(len(band_value.mean_in_pxs_by_ch))),
                        y=band_value.mean_in_pxs_by_ch,
                        line=dict(color=f"rgba({','.join(color_tup)}, 100)")
                    ),
                    row=row_i + 1, col=col_i + 1
                )
                fig.add_trace(
                    go.Scatter(
                        name=group_key,
                        legendgroup=group_key,
                        showlegend=row_i == 0,
                        # x, then x reversed
                        x=[*list(range(len(band_value.mean_in_pxs_by_ch))),
                           *list(range(len(band_value.mean_in_pxs_by_ch)))[::-1]],
                        mode="markers+lines",
                        fill='toself',
                        line=dict(color=f"rgba({','.join(color_tup)}, 0)"),
                        # upper, then lower reversed
                        y=[*band_value.left_dev_in_pxs_by_ch,
                           *band_value.right_dev_in_pxs_by_ch[::-1]]
                    ),
                    row=row_i + 1, col=col_i + 1
                )
            else:
                raise Exception("Undefined mode")
    if x_range is not None: fig.update_xaxes(range=x_range)
    if y_range is not None: fig.update_yaxes(range=y_range)

    fig.update_layout(height=max_snapshots_in_class * 400, width=2500,
                      title_text="classes reflectance comparison")
    fig.write_html(res_path)


class Feature:
    def __init__(self, x_key: str, y_key: str, x_title: str, y_title: str, title: str):
        self.x_key = x_key
        self.y_key = y_key
        self.x_title = x_title
        self.y_title = y_title
        self.title = title


# CLASS_COLORS = ['#4FD51D', '#FF9999', '#E70000', "#830000", "#180000"]
CLASS_COLORS = ['#4FD51D', '#FF9999', '#E70000', '#2ED30B', '#DD7777', '#C50000']


def draw_snapshots_in_features_space(features_df: pd.DataFrame, res_dir: Path):
    features_list: List[Feature] = [
        # Feature(
        #     x_key='mean_agg_in_channels',
        #     x_title='Mean value of channels',
        #     y_key='dev_agg_in_channels',
        #     y_title='Mean deviation of channels',
        #     title='comparison of snapshots aggregated in channels'
        # ),
        Feature(
            x_key='mean_agg_in_pixels',
            x_title='Mean value of pixels',
            y_key='dev_agg_in_pixels',
            y_title='Mean deviation of channels',
            title='comparison of snapshots aggregated in pixels'
        ),
        Feature(
            x_key='too_low_pxs_mean',
            x_title='Mean value of lowest pixels',
            y_key='too_high_pxs_mean',
            y_title='Mean value of highest pixels',
            title='comparison of snapshots aggregated in pixels by lowest and highets pixels'
        ),
        Feature(
            x_key='cl_low_part',
            x_title='Normalized size of cold cluster',
            y_key='cl_all_het',
            y_title='Heterogeneity difference between two clusters',
            title='comparison of snapshots aggregated in pixels by lowest and highets pixels'
        ),
        Feature(
            x_key='cl_low_het',
            x_title='Heterogeneity of cold cluster',
            y_key='cl_high_het',
            y_title='Heterogeneity of hot cluster',
            title='comparison of snapshots aggregated in pixels by lowest and highets pixels'
        )
    ]

    for band_name in BANDS_DICT.keys():
        titles_band = [f"{feature.title} in {band_name} with logreg f1(phyto) = " \
                       f"{clf_build(features_df=features_df, x_keys=[f'{band_name}_{feature.x_key}', f'{band_name}_{feature.y_key}'], y_key='class_generalized')['f1_phyto']}"
                       for feature in features_list]

        fig = make_subplots(rows=1, cols=len(features_list), subplot_titles=titles_band)

        for class_name, color in zip(list(features_df['class'].unique()), CLASS_COLORS):
            for i, feature in enumerate(features_list):
                fig.add_trace(
                    go.Scatter(
                        name=class_name,
                        legendgroup=class_name,
                        showlegend=i == 0,
                        mode='markers',
                        text=features_df[features_df['class'] == class_name]['name'],
                        x=features_df[features_df['class'] == class_name][f"{band_name}_{feature.x_key}"],
                        y=features_df[features_df['class'] == class_name][f"{band_name}_{feature.y_key}"],
                        marker=dict(
                            color=color,
                            size=20,
                            line=dict(
                                color='Black',
                                width=2
                            )
                        ),
                    ),
                    row=1, col=i + 1
                )
                fig['layout'][f'xaxis{i + 1}']['title'] = f"{feature.x_title} in {band_name}"
                fig['layout'][f'yaxis{i + 1}']['title'] = f"{feature.y_title} in {band_name}"

        fig.update_layout(height=1200, width=1200 * len(features_list))
        fig.write_html(f"{res_dir}/{band_name}/features_space_in_band={band_name}.html")


def draw_snapshots_in_all_paired_features_space(features_df: pd.DataFrame, res_dir: Path):
    for band_name in BANDS_DICT.keys():
        features_list = [feature_name for feature_name in list(features_df.keys()) if band_name in feature_name]

        if len(features_list) == 0:
            continue

        fig, axes = plt.subplots(nrows=len(features_list), ncols=len(features_list), figsize=(25, 25))
        for class_name, color in zip(list(features_df['class'].unique()), CLASS_COLORS):
            for j, feature_x in enumerate(features_list):
                for i, feature_y in enumerate(features_list):
                    axes[j][i].scatter(
                        x=features_df[features_df['class'] == class_name][feature_x],
                        y=features_df[features_df['class'] == class_name][feature_y],
                        color=color
                    )
                    axes[j][i].set_ylabel(feature_y)
                    axes[j][i].set_xlabel(feature_x)

        fig.tight_layout()
        fig.savefig(f"{res_dir}/{band_name}/features_space_pairs_in_band={band_name}.png")


def draw_tsne(features_df: pd.DataFrame, features: List[str], res_dir: Optional[str] = None):
    tsne_arr = TSNE().fit_transform(features_df[features])

    fig = go.Figure()

    for class_name, color in zip(list(features_df['class'].unique()), CLASS_COLORS):
        class_rows = features_df[features_df['class'] == class_name]
        fig.add_trace(
            go.Scatter(
                name=class_name,
                legendgroup=class_name,
                mode='markers',
                text=class_rows['name'],
                x=tsne_arr[class_rows.index, 0],
                y=tsne_arr[class_rows.index, 1],
                marker=dict(
                    color=color,
                    size=20,
                    line=dict(
                        color='Black',
                        width=2
                    )
                ),
            )
        )

    if res_dir is None:
        fig.show()
    else:
        fig.write_html(f"{res_dir}/features_in_tsne.html")


def draw_hp_glasses(
        all_classes: List[List[SnapshotMeta]],
        classes_names: List[str],
        bname: str,
        res_path: str
):
    assert len(all_classes) == len(classes_names)

    all_wl_lists = [len(snapshot.bands[bname].wave_lengths)
                    for class_snapshots in all_classes for snapshot in class_snapshots]
    assert all([wl_length == all_wl_lists[0] for wl_length in all_wl_lists])
    wl_lengths = all_classes[0][0].bands[bname].wave_lengths

    max_snap_width = max([snapshot.bands[bname].coordinates[:, 0].max() - snapshot.bands[bname].coordinates[:, 0].min()
                          for class_snapshots in all_classes for snapshot in class_snapshots]) + 5

    max_snap_height = max([snapshot.bands[bname].coordinates[:, 1].max() - snapshot.bands[bname].coordinates[:, 1].min()
                           for class_snapshots in all_classes for snapshot in class_snapshots]) + 5

    fig = go.Figure()

    for wl_id, wl in enumerate(wl_lengths):
        for col_i, class_snapshots in enumerate(all_classes[::-1]):
            for row_i, snapshot in enumerate(class_snapshots):
                norm_x = snapshot.bands[bname].coordinates[:, 0] - min(snapshot.bands[bname].coordinates[:, 0])
                norm_y = snapshot.bands[bname].coordinates[:, 1] - min(snapshot.bands[bname].coordinates[:, 1])
                fig.add_trace(
                    go.Heatmap(
                        visible=False,
                        # z=snapshot.bands['all'].get_too_low_pxs(wl=wl)[:,0],
                        z=snapshot.bands[bname].get_band_data_in_ch_id(ch_id=wl_id),
                        x=norm_x + row_i * max_snap_width,
                        y=norm_y + col_i * max_snap_height,
                        hoverongaps=False
                    )
                )
                if wl_id == 0:
                    fig.add_annotation(
                        x=(norm_x + row_i * max_snap_width).max(),
                        y=(norm_y + col_i * max_snap_height).max(),
                        xref="x",
                        yref="y",
                        text=f"{snapshot.name}/"
                             f"[mean_pixel={snapshot.bands[bname].mean_in_pxs_by_ch.mean().round(2)}, "
                             f"mean_dev={snapshot.bands[bname].mean_dev_in_px.round(2)}, "
                             f"{snapshot.bands[bname].cl_features}]"
                        # f"[lowest_avg={snapshot.bands[bname].get_too_low_pxs().mean().round(2)}, "
                        # f"highest_avg={snapshot.bands[bname].get_too_high_pxs().mean().round(2)}]"
                        ,
                        font=dict(
                            family="Courier New, monospace",
                            size=16,
                            color="#ffffff"
                        ),
                        align="center",
                        bordercolor="#c7c7c7",
                        borderwidth=2,
                        borderpad=4,
                        bgcolor="#ff7f0e",
                        opacity=0.8
                    )

    for col_i, class_snapshots in enumerate(all_classes):
        for row_i, snapshot in enumerate(class_snapshots):
            fig.data[0 + col_i * len(class_snapshots) + row_i].visible = True

    steps = []
    for wl_id, wl in enumerate(wl_lengths):
        step = {
            "method": "update",
            "args": [
                {"visible": [False] * len(fig.data)},
                {"title": "Slider switched to wl: " + str(wl)}
            ],  # layout attribute
        }
        for col_i, class_snapshots in enumerate(all_classes):
            for row_i, snapshot in enumerate(class_snapshots):
                # Toggle i'th trace to "visible"
                step["args"][0]["visible"][wl_id + col_i * len(class_snapshots) + row_i] = True

        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig.update_coloraxes(
        showscale=False,
        # colorbar=dict(showticklabels=False)
    )

    fig.write_html(res_path)
