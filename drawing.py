from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.manifold import TSNE

from clf import clf_build
from snapshots_processing import SnapshotMeta, BANDS_DICT
import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'Times New Roman',
        'size': 22}
matplotlib.rc('font', **font)


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
                        # px, then px reversed
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
                        # px, then px reversed
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


def draw_points_tsne(pt_groups: List[np.ndarray], groups_draw_args):
    assert len(pt_groups) == len(groups_draw_args)
    all_features = np.vstack(pt_groups)
    all_tsne = TSNE(perplexity=min(100, len(all_features) - 1)).fit_transform(all_features)
    pt_groups_tsne = []
    lb = 0
    for pt_group in pt_groups:
        pt_groups_tsne.append(all_tsne[lb:lb + len(pt_group)])
        lb += len(pt_group)

    for pt_group_tsne, group_draw_args in zip(pt_groups_tsne, groups_draw_args):
        plt.scatter(pt_group_tsne[:, 0], pt_group_tsne[:, 1], **group_draw_args)

    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.tight_layout()


class Feature:
    def __init__(self, x_key: str, y_key: str, x_title: str, y_title: str, title: str):
        self.x_key = x_key
        self.y_key = y_key
        self.x_title = x_title
        self.y_title = y_title
        self.title = title


# CLASS_COLORS = ['#4FD51D', '#FF9999', '#E70000', "#830000", "#180000"]

# POTATO & POTATO NEW
# CLASS_COLORS = ['#4FD51D', '#FF9999', '#E70000', '#2ED30B', '#DD7777', '#C50000']

# DIFFERENT WHEAT
CLASS_COLORS = ['#88FF6F', '#FF9999', '#2ED30B', '#DD7777', '#41C426', '#D07272']


# CLASS_COLORS = ['#4FD51D', '#FF9999', '#4FD51D', '#FF9999']


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
                       f"{clf_build(fit_df=features_df, eval_df=features_df, x_keys=[f'{band_name}_{feature.x_key}', f'{band_name}_{feature.y_key}'], y_key='class_generalized')['test_f1_phyto']}"
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


RENAME_DICT = {
    'all_pixels_mean': 'MEAN',
    'all_pixels_std': 'DEV',

    'too_low_pxs_mean': 'LOW MEAN',
    'too_low_pxs_std': 'LOW STD',

    'too_high_pxs_mean': 'HIGH MEAN',
    'too_high_pxs_std': 'HIGH STD',
}


def rename_axis(name: str) -> str:
    split = name.split('_')
    ch_name = split[0]
    func_name = RENAME_DICT['_'.join(split[1:])]
    return f"{func_name}({ch_name})".upper()


def draw_snapshots_in_paired_features_space(features_df: pd.DataFrame, features_list: List[str], res_dir: Path,
                                            fname: str):
    each_fig_size = 2.5

    fig, axes = plt.subplots(
        nrows=len(features_list), ncols=len(features_list),
        figsize=(each_fig_size * len(features_list), each_fig_size * len(features_list))
    )
    for class_name, color in zip(list(features_df['class'].unique()), CLASS_COLORS):
        for j, feature_x in enumerate(features_list):
            for i, feature_y in enumerate(features_list):
                axes[j][i].scatter(
                    x=features_df[features_df['class'] == class_name][feature_x],
                    y=features_df[features_df['class'] == class_name][feature_y],
                    color=color
                )
                if 'BGI' not in features_list:  # great topic bone
                    axes[j][i].set_ylabel(rename_axis(feature_y))
                    axes[j][i].set_xlabel(rename_axis(feature_x))
                else:
                    axes[j][i].set_ylabel(feature_y)
                    axes[j][i].set_xlabel(feature_x)

    fig.tight_layout()
    fig.savefig(f"{res_dir}/{fname}")


def draw_snapshots_in_all_paired_features_space(features_df: pd.DataFrame, res_dir: Path):
    for band_name in BANDS_DICT.keys():
        print(f"draw all paired features in {band_name}")

        features_list = [feature_name for feature_name in list(features_df.keys())
                         if band_name == feature_name.split('_')[0]]

        features_list = [feature for feature in features_list if feature in features_df.keys()]

        if len(features_list) == 0:
            break

        draw_snapshots_in_paired_features_space(
            features_df=features_df, features_list=features_list,
            res_dir=res_dir.joinpath(band_name), fname=f'features_space_pairs_in_band={band_name}.png'
        )

    indexes = ['ARI', 'BGI', 'BRI', 'CRI1', 'CRI2', 'CSI1', 'CSI2', 'CUR', 'gNDVI', 'hNDVI', 'NPCI']
    draw_snapshots_in_paired_features_space(
        features_df=features_df, features_list=indexes,
        res_dir=res_dir, fname='features_space_pairs_indexes.png'
    )


def draw_tsne_plotly(features_df: pd.DataFrame, features: List[str], save_pref: Optional[str] = None):
    tsne_arr = TSNE().fit_transform(features_df[features])

    fig = go.Figure()
    features_df = features_df.reset_index(inplace=False)
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

    fig.update_layout(height=800, width=1000, title_text="t-SNE representation")

    if save_pref is None:
        fig.show()
    else:
        fig.write_html(f"{save_pref}.html")


CLASS_NAME_MAP = {
    'health2': 'health1',
    'puccinia phyto2': 'puccinia 1',
    'health3': 'health2',
    'puccinia phyto3': 'puccinia 2',
}


def draw_tsne_matplot(features_df: pd.DataFrame, features: List[str], save_pref: Optional[str] = None):
    tsne_arr = TSNE(perplexity=100).fit_transform(features_df[features])

    plt.rcParams["figure.figsize"] = (10, 10)

    features_df = features_df.reset_index(inplace=False)
    for class_name, color in zip(list(features_df['class'].unique()), CLASS_COLORS):
        curr_class_df = features_df[features_df['class'] == class_name]
        # class_centroid = curr_class_df[features].mean().reset_index()[0]
        # class_distances = [np.linalg.norm(class_centroid - np.array(tup)) for tup in
        #                    curr_class_df[features].itertuples(index=False)]
        plt.scatter(
            tsne_arr[curr_class_df.index, 0],
            tsne_arr[curr_class_df.index, 1],
            label=CLASS_NAME_MAP[class_name],
            color=color,
            edgecolor="black",
            s=200
        )
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    # plt.legend(loc="upper right")
    plt.legend()
    plt.tight_layout()

    if save_pref is None:
        plt.show()
    else:
        plt.savefig(f"{save_pref}.png", dpi=100)
        plt.clf()


def draw_hp_glasses(
        all_classes: List[List[SnapshotMeta]],
        classes_names: List[str],
        bname: str,
        res_path: str
):
    print(f'drawing hp_glasses {res_path}')
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

    draw_class_names = ['health ds1', 'phyto ds1', 'health ds2', 'phyto ds 2']
    draw_class_names = list(reversed(draw_class_names))

    for wl_id, wl in enumerate(wl_lengths):
        for class_i, class_snapshots in enumerate(all_classes[::-1]):
            for snap_i, snapshot in enumerate(class_snapshots):
                norm_x = snapshot.bands[bname].coordinates[:, 0] - min(snapshot.bands[bname].coordinates[:, 0])
                norm_y = snapshot.bands[bname].coordinates[:, 1] - min(snapshot.bands[bname].coordinates[:, 1])
                fig.add_trace(
                    go.Heatmap(
                        visible=False,
                        # z=snapshot.bands['all'].get_too_low_pxs(wl=wl)[:,0],
                        z=snapshot.bands[bname].get_band_data_in_ch_id(ch_id=wl_id),
                        x=norm_x + snap_i * max_snap_width,
                        y=norm_y + class_i * max_snap_height,
                        hoverongaps=False
                    )
                )
                if wl_id == 0 and snap_i == 0:
                    fig.add_annotation(
                        # px=(norm_x + snap_i * max_snap_width).max(),
                        x=(-10 + snap_i * max_snap_width).max(),
                        y=(norm_y + class_i * max_snap_height).max(),
                        xref="px",
                        yref="py",
                        # text=f"{snapshot.name}/"
                        text=draw_class_names[class_i]
                        # f"[mean_pixel={snapshot.bands[bname].mean_in_pxs_by_ch.mean().round(2)}, "
                        # f"mean_dev={snapshot.bands[bname].mean_dev_in_px.round(2)}, "
                        # f"{snapshot.bands[bname].cl_features}]"
                        # f"[lowest_avg={snapshot.bands[bname].get_too_low_pxs().mean().round(2)}, "
                        # f"highest_avg={snapshot.bands[bname].get_too_high_pxs().mean().round(2)}]"
                        ,
                        font=dict(
                            family="Times New Roman",
                            size=30,
                            color="#ffffff"
                        ),
                        align="center",
                        bordercolor="#c7c7c7",
                        borderwidth=2,
                        borderpad=4,
                        bgcolor=CLASS_COLORS[(class_i + 1) % 2],
                        opacity=0.8
                    )

    for class_i, class_snapshots in enumerate(all_classes):
        for snap_i, snapshot in enumerate(class_snapshots):
            fig.data[0 + class_i * len(class_snapshots) + snap_i].visible = True

    steps = []
    for wl_id, wl in enumerate(wl_lengths):
        step = {
            "method": "update",
            "args": [
                {"visible": [False] * len(fig.data)},
                {"title": "Slider switched to wl: " + str(wl)}
            ],  # layout attribute
        }
        for class_i, class_snapshots in enumerate(all_classes):
            for snap_i, snapshot in enumerate(class_snapshots):
                # Toggle i'th trace to "visible"
                step["args"][0]["visible"][wl_id + class_i * len(class_snapshots) + snap_i] = True

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


def draw_n_vectors_matplot(
        x_list: List[List[float]],
        y_list: List[List[float]],
        meta_list: List[List[float]],
        labels: List[str],
        x_label: str,
        y_label: str,
        title: str,
        save_pref=None
):
    for x_arr, y_arr, label in zip(x_list, y_list, labels):
        plt.plot(x_arr, y_arr, label=label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="lower right")
        plt.title(title)

    plt.tight_layout()

    if save_pref is None:
        plt.show()
    else:
        plt.savefig(f"{save_pref}.png", dpi=100)
        plt.clf()


def draw_n_vectors_plotly(
        x_list: List[List[float]],
        y_list: List[List[float]],
        meta_list: List[List[float]],
        labels: List[str],
        x_label: str,
        y_label: str,
        title: str,
        save_pref=None
):
    fig = go.Figure()

    for x_arr, y_arr, meta_arr, label in zip(x_list, y_list, meta_list, labels):
        fig.add_trace(
            go.Scatter(
                name=label,
                legendgroup=label,
                text=meta_arr,
                x=x_arr,
                y=y_arr,
            ),
        )

        fig.update_xaxes(title_font_family=x_label)
        fig.update_yaxes(title_font_family=y_label)
        fig.update_layout(height=800, width=1000, title_text=title)

        if save_pref is None:
            fig.show()
        else:
            fig.write_html(f"{save_pref}.html")
