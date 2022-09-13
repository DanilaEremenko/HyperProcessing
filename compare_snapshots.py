from pathlib import Path
from typing import List, Dict, Optional

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


RES_DIR = Path('comparison_all')
RES_DIR.mkdir(exist_ok=True)


class BandData:
    def get_left_confidence_interval_pixels(self, band_data: np.ndarray, left_part: float) -> np.ndarray:
        assert 0 <= left_part <= 1
        all_mean_agg_in_px_by_ch = band_data.mean(axis=1)
        df = pd.DataFrame({
            'pix_id': list(range(len(all_mean_agg_in_px_by_ch))),
            'pix_mean': all_mean_agg_in_px_by_ch
        }).sort_values(['pix_mean']).iloc[:int(len(all_mean_agg_in_px_by_ch) * left_part)]

        return band_data[list(df['pix_id'])]

    def get_pixels_by_ch_id(self, ch_id: int) -> np.ndarray:
        return self.band_data[:, ch_id]

    def get_pixels_by_wl(self, target_wl: int) -> np.ndarray:
        ch_id = [i for i, wl in enumerate(self.wave_lengths) if wl == target_wl][0]
        return self.get_pixels_by_ch_id(ch_id=ch_id)

    def get_band_corr_df(self) -> pd.DataFrame:
        return pd.DataFrame({str(wl): self.band_data[:, i] for i, wl in enumerate(self.wave_lengths)}).corr()

    def __init__(self, left_wl_bound: int, right_wl_bound: int, all_data: np.ndarray):
        # separate coordinates and snapshot data
        self.coordinates = all_data[1:, :2]

        snapshot = all_data[:, 2:]

        wl_ids = [wl_id for wl_id, wl in enumerate(snapshot[0]) if left_wl_bound < wl < right_wl_bound]
        assert len(wl_ids) > 0
        self.wave_lengths = snapshot[0, wl_ids]

        # filter wavelengths
        band_data = snapshot[1:, wl_ids]
        # filter pixels
        # band_data = self.get_left_confidence_interval_pixels(band_data, left_part=0.05)

        self.mean_agg_in_ch_by_px = band_data.mean(axis=0)
        self.left_dev_agg_in_ch_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='left')
        self.right_dev_in_ch_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='right')

        self.mean_agg_in_px_by_ch = band_data.mean(axis=1)
        self.left_dev_agg_in_px_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='left')
        self.right_dev_agg_in_px_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='right')

        self.band_data = band_data


BANDS_DICT = {
    # 'blue': (440, 485), 'cyan': (485, 500), 'green': (500, 565), 'yellow': (565, 590),
    # 'orange': (590, 625), 'red': (625, 780),
    # 'visible': (440, 780),
    # 'infrared': (780, 1000),
    'all': (400, 1000)
}


class SnapshotMeta:
    def __init__(self, name: str, all_data: np.ndarray):
        self.name = name
        self.bands: Dict[str, BandData] = {
            band_name: BandData(left_wl_bound=band_range[0], right_wl_bound=band_range[1], all_data=all_data)
            for band_name, band_range in BANDS_DICT.items()
        }


def parse_classes(classes_dict: Dict[str, List[str]], max_files_in_dir: int) -> Dict[str, List[SnapshotMeta]]:
    classes_features: Dict[str, List[SnapshotMeta]] = {class_name: [] for class_name in classes_dict.keys()}
    for class_name, class_dirs in classes_dict.items():
        features = classes_features[class_name]
        for dir_path in class_dirs:
            files = list(os.listdir(dir_path))[:max_files_in_dir]
            for file_id, file in enumerate(files):
                all_data = genfromtxt(
                    f'{dir_path}/{file}',
                    delimiter=','
                )
                features.append(
                    SnapshotMeta(name=file, all_data=all_data)
                )
    return classes_features


def draw_snapshots_as_reflectance(classes_dict: Dict[str, List[SnapshotMeta]], res_path: str,
                                  x_range: Optional[tuple] = None, y_range: Optional[tuple] = None, mode='ch'):
    for coord_range in (x_range, y_range): assert len(coord_range) == 2

    max_snapshots_in_class = max([len(snapshots) for snapshots in classes_dict.values()])
    fig = make_subplots(
        rows=max_snapshots_in_class,
        cols=len(classes_dict)
    )
    colors = [('0', '180', '0'), ('255', '153', '153'), ('250', '0', '0'), ('113', '12', '12'), ('10', '10', '10')]

    assert len(colors) >= len(classes_dict)

    for col_i, ((group_key, group_list), color_tup) in enumerate(zip(classes_dict.items(), colors)):
        for row_i, snapshot_meta in enumerate(group_list):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta
            for band_i, (band_key, band_value) in enumerate(snapshot_meta.bands.items()):
                band_value: BandData = band_value
                if mode == 'ch':
                    # vector of wavelengths representation
                    fig.add_trace(
                        go.Scatter(
                            name=group_key,
                            legendgroup=group_key,
                            showlegend=row_i == 0 and band_i == 0,
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
                            showlegend=row_i == 0 and band_i == 0,
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
                elif mode == 'px':
                    fig.add_trace(
                        go.Scatter(
                            name=group_key,
                            legendgroup=group_key,
                            showlegend=row_i == 0 and band_i == 0,
                            x=list(range(len(band_value.mean_agg_in_px_by_ch))),
                            y=band_value.mean_agg_in_px_by_ch,
                            line=dict(color=f"rgba({','.join(color_tup)}, 100)")
                        ),
                        row=row_i + 1, col=col_i + 1
                    )
                    fig.add_trace(
                        go.Scatter(
                            name=group_key,
                            legendgroup=group_key,
                            showlegend=row_i == 0 and band_i == 0,
                            # x, then x reversed
                            x=[*list(range(len(band_value.mean_agg_in_px_by_ch))),
                               *list(range(len(band_value.mean_agg_in_px_by_ch)))[::-1]],
                            mode="markers+lines",
                            fill='toself',
                            line=dict(color=f"rgba({','.join(color_tup)}, 0)"),
                            # upper, then lower reversed
                            y=[*band_value.left_dev_agg_in_px_by_ch,
                               *band_value.right_dev_agg_in_px_by_ch[::-1]]
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


def get_features_df(group_features: Dict[str, List[SnapshotMeta]]) -> pd.DataFrame:
    band_metrics = [
        'mean_agg_in_channels_sum',
        'dev_agg_in_channels_sum',

        'mean_agg_in_pixels_sum',
        'dev_agg_in_pixels_sum',
        'left_dev_agg_in_pixels_sum'
    ]

    features_dict = {
        **{f"{band_name}_{band_metric}": [] for band_name in BANDS_DICT.keys() for band_metric in band_metrics},
        'class': [],
        'name': []
    }

    for col_i, (group_key, group_list) in enumerate(group_features.items()):
        for row_i, snapshot_meta in enumerate(group_list):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta

            for band_key, band_value in snapshot_meta.bands.items():
                # cast for ide
                band_value: BandData = band_value

                features_dict[f'{band_key}_mean_agg_in_channels_sum'].append(
                    band_value.mean_agg_in_ch_by_px.sum()
                )
                features_dict[f'{band_key}_dev_agg_in_channels_sum'].append(
                    band_value.right_dev_in_ch_by_px.sum() - band_value.left_dev_agg_in_ch_by_px.sum()
                )

                features_dict[f'{band_key}_mean_agg_in_pixels_sum'].append(
                    band_value.mean_agg_in_px_by_ch.sum()
                )
                features_dict[f'{band_key}_dev_agg_in_pixels_sum'].append(
                    band_value.right_dev_agg_in_px_by_ch.sum() - band_value.left_dev_agg_in_px_by_ch.sum()
                )

                features_dict[f'{band_key}_left_dev_agg_in_pixels_sum'].append(
                    band_value.left_dev_agg_in_px_by_ch.sum()
                    # band_value.mean_agg_in_px_by_ch
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
        title: str,
        colors: List[str],
        res_path: str
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
        title_text=title
    )
    fig.write_html(res_path)


def draw_detailed_comparison(
        all_classes: List[List[SnapshotMeta]],
        classes_names: List[str],
        res_path: str
):
    assert len(all_classes) == len(classes_names)

    all_wl_lists = [len(snapshot.bands['all'].wave_lengths)
                    for class_snapshots in all_classes for snapshot in class_snapshots]
    assert all([wl_length == all_wl_lists[0] for wl_length in all_wl_lists])
    wl_lengths = all_classes[0][0].bands['all'].wave_lengths

    max_snap_width = max([snapshot.bands['all'].coordinates[:, 0].max() - snapshot.bands['all'].coordinates[:, 0].min()
                          for class_snapshots in all_classes for snapshot in class_snapshots]) + 5

    max_snap_height = max([snapshot.bands['all'].coordinates[:, 1].max() - snapshot.bands['all'].coordinates[:, 1].min()
                           for class_snapshots in all_classes for snapshot in class_snapshots]) + 5

    fig = go.Figure()

    for wl_id, _ in enumerate(wl_lengths):
        for col_i, class_snapshots in enumerate(all_classes[::-1]):
            for row_i, snapshot in enumerate(class_snapshots):
                norm_x = snapshot.bands['all'].coordinates[:, 0] - min(snapshot.bands['all'].coordinates[:, 0])
                norm_y = snapshot.bands['all'].coordinates[:, 1] - min(snapshot.bands['all'].coordinates[:, 1])
                fig.add_trace(
                    go.Heatmap(
                        visible=False,
                        z=snapshot.bands['all'].get_pixels_by_ch_id(ch_id=wl_id),
                        x=norm_x + row_i * max_snap_width,
                        y=norm_y + col_i * max_snap_height,
                        hoverongaps=False
                    )
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


def draw_files(classes_dict: Dict[str, List[str]], max_files_in_dir: int):
    classes_features_dict = parse_classes(classes_dict=classes_dict, max_files_in_dir=max_files_in_dir)

    draw_detailed_comparison(
        all_classes=[classes_features_dict[key][0:2] for key in classes_features_dict.keys()],
        classes_names=[key for key in classes_features_dict.keys()],
        res_path=f'{RES_DIR}/classes_comparison_by_features.html'
    )

    draw_snapshots_as_reflectance(classes_features_dict, res_path=f'{RES_DIR}/comparison_by_agg_in_channels.html',
                                  x_range=(0, 900), y_range=(0, 10_000), mode='ch')
    draw_snapshots_as_reflectance(classes_features_dict, res_path=f'{RES_DIR}/comparison_by_agg_in_pixels.html',
                                  x_range=(0, 200), y_range=(8_000, 10_000), mode='px')

    features_df = get_features_df(group_features=classes_features_dict)
    for band_name in BANDS_DICT.keys():
        draw_snapshots_as_features(
            features_df=features_df,
            x_key=f'{band_name}_mean_agg_in_channels_sum',
            y_key=f'{band_name}_dev_agg_in_channels_sum',
            x_title=f'Area under mean curve aggregated by pixels in in channels in {band_name} range',
            y_title='Area between right deviation and left deviation aggregated by pixels in channels',
            title="comparison of snapshots aggregated in channels",
            colors=['#4FD51D', '#FF9999', '#E70000', "#830000", "#180000"],
            res_path=f'{RES_DIR}/{band_name}_comparison_by_features_agg_in_channels.html'
        )

        draw_snapshots_as_features(
            features_df=features_df,
            x_key=f'{band_name}_mean_agg_in_pixels_sum',
            y_key=f'{band_name}_dev_agg_in_pixels_sum',
            x_title=f'Area under mean curve aggregated by channels in pixels in {band_name} range',
            y_title='Area between right deviation and left deviation aggregated by channels in pixels',
            title="comparison of snapshots aggregated in pixels",
            colors=['#4FD51D', '#FF9999', '#E70000', "#830000", "#180000"],
            res_path=f'{RES_DIR}/{band_name}_comparison_by_features_agg_in_pixels.html'
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
