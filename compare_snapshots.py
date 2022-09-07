from typing import List, Dict

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd


def draw_klebs_np(ax, snapshot: np.ndarray, title: str):
    # for i in range(0, len(snapshot) - 1, 10):
    #     plt.plot(snapshot[0], snapshot[i + 1])
    ax.plot(snapshot[0], snapshot[1:].mean(axis=0))
    ax.set_xlim(400, 900)
    ax.set_ylim(0, 10_000)
    ax.set_title(title)


def calculate_dev_in_channels(arr, mode='left') -> np.ndarray:
    residuals = arr - arr.mean(axis=0)
    ch_num = residuals.shape[1]
    dev_arr = np.zeros(ch_num)
    if mode == 'left':
        for ch_id in range(ch_num):
            res_in_ch = residuals[:, ch_id]
            dev_arr[ch_id] = res_in_ch[res_in_ch < 0].mean()
    elif mode == 'right':
        for ch_id in range(ch_num):
            res_in_ch = residuals[:, ch_id]
            dev_arr[ch_id] = res_in_ch[res_in_ch > 0].mean()

    return dev_arr


def calculate_dev_in_pixels(arr, mode='left') -> np.ndarray:
    residuals = arr - np.expand_dims(arr.mean(axis=1), axis=1)
    pixels_num = residuals.shape[0]
    dev_arr = np.zeros(pixels_num)
    if mode == 'left':
        for pix_id in range(pixels_num):
            res_in_ch = residuals[pix_id]
            dev_arr[pix_id] = res_in_ch[res_in_ch < 0].mean()
    elif mode == 'right':
        for pix_id in range(pixels_num):
            res_in_ch = residuals[pix_id]
            dev_arr[pix_id] = res_in_ch[res_in_ch > 0].mean()

    return dev_arr


class BandData:
    def __init__(self, left_wl_bound: int, right_wl_bound: int, snapshot: np.ndarray):
        wl_ids = [wl_id for wl_id, wl in enumerate(snapshot[0]) if left_wl_bound < wl < right_wl_bound]
        assert len(wl_ids) > 0
        self.wave_lengths = snapshot[0, wl_ids]
        band_data = snapshot[1:, wl_ids]

        self.mean_agg_in_ch_by_px = band_data.mean(axis=0)
        self.left_dev_agg_in_ch_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='left')
        self.right_dev_in_ch_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='right')

        self.mean_agg_in_px_by_ch = band_data.mean(axis=1)
        self.left_dev_agg_in_px_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='left')
        self.right_dev_agg_in_px_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='right')


BAND_NAMES = ['red', 'infrared']


class SnapshotMeta:
    def __init__(self, name: str, snapshot: np.ndarray):
        # crop x,y
        snapshot = snapshot[:, 2:]

        self.name = name
        self.bands = {
            'red': BandData(left_wl_bound=625, right_wl_bound=740, snapshot=snapshot),
            'infrared': BandData(left_wl_bound=780, right_wl_bound=1000, snapshot=snapshot)
        }

        assert all([key in BAND_NAMES for key in self.bands.keys()])


def parse_classes(classes_dict: Dict[str, List[str]], max_files_in_dir: int) -> Dict[str, List[SnapshotMeta]]:
    classes_features: Dict[str, List[SnapshotMeta]] = {class_name: [] for class_name in classes_dict.keys()}
    for class_name, class_dirs in classes_dict.items():
        features = classes_features[class_name]
        for dir_path in class_dirs:
            files = list(os.listdir(dir_path))[:max_files_in_dir]
            for file_id, file in enumerate(files):
                snapshot = genfromtxt(
                    f'{dir_path}/{file}',
                    delimiter=','
                )
                features.append(
                    SnapshotMeta(name=file, snapshot=snapshot)
                )
    return classes_features


def draw_snapshots_as_reflectance(group_features: Dict[str, List[SnapshotMeta]]):
    max_snapshots_in_class = max([len(snapshots) for snapshots in group_features.values()])
    fig = make_subplots(
        rows=max_snapshots_in_class,
        cols=len(group_features)
    )
    colors = [('0', '180', '0'), ('180', '0', '0')]
    for col_i, ((group_key, group_list), color_tup) in enumerate(zip(group_features.items(), colors)):
        for row_i, snapshot_meta in enumerate(group_list):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta
            for band_key, band_value in snapshot_meta.bands.items():
                fig.add_trace(
                    go.Scatter(
                        name=group_key,
                        legendgroup=group_key,
                        showlegend=row_i == 0,
                        x=band_value.wave_lengths,
                        y=band_value.mean_agg_in_ch_by_px,
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
                        y=[*band_value.left_dev_agg_in_ch_by_px,
                           *band_value.right_dev_in_ch_by_px[::-1]]
                    ),
                    row=row_i + 1, col=col_i + 1
                )
    fig.update_xaxes(range=[700, 900])
    fig.update_yaxes(range=[8_000, 10_000])
    fig.update_layout(height=max_snapshots_in_class * 400, width=1800, title_text="classes reflectance comparison")
    fig.write_html("comparison_by_reflectance.html")


def get_features_df(group_features: Dict[str, List[SnapshotMeta]]) -> pd.DataFrame:
    band_metrics = [
        'mean_infrared_agg_by_pixels_sum',
        'dev_infrared_agg_by_pixels_sum',

        'mean_infrared_agg_by_channels_sum',
        'dev_infrared_agg_by_channels_sum'
    ]

    features_dict = {
        **{f"{band_name}_{band_metric}": [] for band_name in BAND_NAMES for band_metric in band_metrics},
        'class': [],
        'name': []
    }

    for col_i, (group_key, group_list) in enumerate(group_features.items()):
        for row_i, snapshot_meta in enumerate(group_list):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta

            for band_key, band_value in snapshot_meta.bands.items():
                features_dict[f'{band_key}_mean_infrared_agg_by_pixels_sum'].append(
                    band_value.mean_agg_in_ch_by_px.sum()
                )
                features_dict[f'{band_key}_dev_infrared_agg_by_pixels_sum'].append(
                    band_value.right_dev_in_ch_by_px.sum() - band_value.left_dev_agg_in_ch_by_px.sum()
                )

                features_dict[f'{band_key}_mean_infrared_agg_by_channels_sum'].append(
                    band_value.mean_agg_in_px_by_ch.sum()
                )
                features_dict[f'{band_key}_dev_infrared_agg_by_channels_sum'].append(
                    band_value.right_dev_agg_in_px_by_ch.sum() - band_value.left_dev_agg_in_px_by_ch.sum()
                )

            features_dict['class'].append(group_key)

            features_dict['name'].append(snapshot_meta.name)

    features_df = pd.DataFrame(features_dict)

    # normalize features
    # for key in features_df.keys():
    #     if key != 'class':
    #         features_df[key] /= features_df[key].max()

    return features_df


def draw_snapshots_as_features(
        features_df: pd.DataFrame,
        x_key: str, y_key: str,
        x_title: str, y_title: str,
        colors: List[str],
        res_file: str
):
    fig = go.Figure()

    for class_name, color in zip(list(features_df['class'].unique()), colors):
        fig.add_trace(
            go.Scatter(
                name=class_name,
                legendgroup=class_name,
                mode='markers',
                text=features_df[features_df['class'] == class_name]['name'],
                x=features_df[features_df['class'] == class_name][x_key],
                y=features_df[features_df['class'] == class_name][y_key],
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

    fig.update_layout(
        height=1280, width=1800,
        xaxis_title=x_title,
        yaxis_title=y_title,
        title_text="classes reflectance comparison"
    )
    fig.write_html(res_file)


def draw_files(classes_dict: Dict[str, List[str]], max_files_in_dir: int):
    classes_features_dict = parse_classes(classes_dict=classes_dict, max_files_in_dir=max_files_in_dir)
    draw_snapshots_as_reflectance(classes_features_dict)

    features_df = get_features_df(group_features=classes_features_dict)
    for band_name in BAND_NAMES:
        draw_snapshots_as_features(
            features_df=features_df,
            x_key=f'{band_name}_mean_infrared_agg_by_pixels_sum',
            y_key=f'{band_name}_dev_infrared_agg_by_pixels_sum',
            x_title='Area under mean curve aggregated by pixels in infrared range',
            y_title='Difference between area under right deviation and left deviation aggregated by pixels',
            colors=['#4FD51D', '#FF9999', '#E70000', "#830000", "#180000"],
            res_file=f'{band_name}_comparison_by_features_agg_by_pixels.html'
        )

        draw_snapshots_as_features(
            features_df=features_df,
            x_key=f'{band_name}_mean_infrared_agg_by_channels_sum',
            y_key=f'{band_name}_dev_infrared_agg_by_channels_sum',
            x_title='Area under mean curve aggregated by channels in infrared range',
            y_title='Difference between area under right deviation and left deviation aggregated by channels',
            colors=['#4FD51D', '#FF9999', '#E70000', "#830000", "#180000"],
            res_file=f'{band_name}_comparison_by_features_agg_by_channels.html'
        )


if __name__ == '__main__':
    draw_files(
        classes_dict={
            'health': [
                'csv/control/gala-control-bp-1_000',
                'csv/control/gala-control-bp-2_000',
                # 'csv/control/gala-control-bp-3_000',
                # 'csv/control/gala-control-bp-4_000',
            ],
            'phyto1': [
                'csv/phytophthora/gala-phytophthora-bp-1_000',
                'csv/phytophthora/gala-phytophthora-bp-5-1_000',
            ],
            'phyto2': [
                'csv/phytophthora/gala-phytophthora-bp-2_000',
                'csv/phytophthora/gala-phytophthora-bp-6-2_000',
            ],
            'phyto3': [
                'csv/phytophthora/gala-phytophthora-bp-3_000',
                'csv/phytophthora/gala-phytophthora-bp-7-3_000',
            ],
            'phyto4': [
                'csv/phytophthora/gala-phytophthora-bp-4_000',
                'csv/phytophthora/gala-phytophthora-bp-8-4_000',
            ],

        },
        max_files_in_dir=10
    )
