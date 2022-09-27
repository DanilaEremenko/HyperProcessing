from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn import preprocessing, tree, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


class BandRange:
    def __init__(self):
        pass


class BandRangeBounds(BandRange):
    def __init__(self, left_bound: float, right_bound: float):
        super().__init__()
        self.lb = left_bound
        self.rb = right_bound


class BandRangeSet(BandRange):
    def __init__(self, wls: List[float]):
        super().__init__()
        self.wls = wls


RES_DIR = Path('comparison_no_filt_try')
RES_DIR.mkdir(exist_ok=True)

BANDS_DICT = {
    # 'blue_set': BandRangeSet(wls=[450]),

    'blue': BandRangeBounds(440, 485),
    'cyan': BandRangeBounds(485, 500),
    'green': BandRangeBounds(500, 565),
    'yellow': BandRangeBounds(565, 590),
    'orange': BandRangeBounds(590, 625),
    'red': BandRangeBounds(625, 780),
    'visible': BandRangeBounds(440, 780),
    'infrared': BandRangeBounds(780, 1000),

    'all': BandRangeBounds(400, 1000),
}


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

    if residuals.min() == residuals.max() == 0:
        if residuals.shape[1] != 1:
            raise Exception("All channels have same value for pixel")
    elif mode == 'left':
        for pix_id in range(pixels_num):
            res_in_ch = residuals[pix_id]
            dev_arr[pix_id] = res_in_ch[res_in_ch < 0].mean()
    elif mode == 'right':
        for pix_id in range(pixels_num):
            res_in_ch = residuals[pix_id]
            dev_arr[pix_id] = res_in_ch[res_in_ch > 0].mean()
    else:
        raise Exception(f"Undefined mode = {mode}")

    return dev_arr


class BandData:
    def get_filtered_interval_pixels(
            self,
            band_data: np.ndarray,
            mode: str,
            part: float,
            fill_out_with_zeros=False
    ) -> np.ndarray:
        assert 0 <= part <= 1
        all_mean_agg_in_px_by_ch = band_data.mean(axis=1)
        df = pd.DataFrame({
            'pix_id': list(range(len(all_mean_agg_in_px_by_ch))),
            'pix_mean': all_mean_agg_in_px_by_ch
        }).sort_values(['pix_mean'])

        if mode == 'save_left':
            df = df.iloc[:int(len(all_mean_agg_in_px_by_ch) * part)]
        elif mode == 'save_right':
            df = df.iloc[-int(len(all_mean_agg_in_px_by_ch) * part):]
        elif mode == 'crop_left':
            df = df.iloc[-int(len(all_mean_agg_in_px_by_ch) * (1 - part)):]
        elif mode == 'crop_right':
            df = df.iloc[:int(len(all_mean_agg_in_px_by_ch) * (1 - part))]
        elif mode == 'crop_edges':
            df = df.iloc[-int(len(all_mean_agg_in_px_by_ch) * (1 - part))
                         :int(len(all_mean_agg_in_px_by_ch) * (1 - part))]
        else:
            raise Exception(f"Unexpected mode = {mode}")

        ok_ids = list(df['pix_id'])
        if fill_out_with_zeros:
            not_ok_ids = [i for i in range(band_data.shape[0]) if i not in ok_ids]
            zeros_filled_band_data = band_data.copy()
            zeros_filled_band_data[not_ok_ids] = 0
            return zeros_filled_band_data
        else:
            return band_data[ok_ids].copy()

    def get_band_data_in_ch_id(self, ch_id: int) -> np.ndarray:
        return self.band_data[:, ch_id]

    def get_band_data_in_wl(self, target_wl: float) -> np.ndarray:
        ch_id = [i for i, wl in enumerate(self.wave_lengths) if wl == target_wl][0]
        return self.get_band_data_in_ch_id(ch_id=ch_id)

    def get_band_corr_df(self) -> pd.DataFrame:
        return pd.DataFrame({str(wl): self.band_data[:, i] for i, wl in enumerate(self.wave_lengths)}).corr()

    def get_too_low_pxs(self) -> np.ndarray:
        return self.get_filtered_interval_pixels(
            # band_data=np.expand_dims(self.get_band_data_in_wl(target_wl=wl), 1),
            band_data=self.band_data,
            mode='save_left',
            part=0.1,
            fill_out_with_zeros=True
        )

    def get_too_high_pxs(self) -> np.ndarray:
        return self.get_filtered_interval_pixels(
            # band_data=np.expand_dims(self.get_band_data_in_wl(target_wl=wl), 1),
            band_data=self.band_data,
            mode='save_right',
            part=0.1,
            fill_out_with_zeros=True
        )

    def get_mid_pxs(self) -> np.ndarray:
        return self.get_filtered_interval_pixels(
            # band_data=np.expand_dims(self.get_band_data_in_wl(target_wl=wl), 1),
            band_data=self.band_data,
            mode='crop_edges',
            part=0.1,
            fill_out_with_zeros=True
        )

    @property
    def mean_dev_in_px(self) -> np.float_:
        return self.right_dev_in_pxs_by_ch.mean() - self.left_dev_in_pxs_by_ch.mean()

    @property
    def mean_dev_in_ch(self) -> np.float_:
        return self.right_dev_in_chs_by_px.mean() - self.left_dev_in_chs_by_px.mean()

    def __init__(self, band_range: BandRange, all_data: np.ndarray):
        # separate coordinates and snapshot data
        self.coordinates = all_data[1:, :2].copy()

        # coordinates matching with image format
        self.coordinates[:, 0] = all_data[1:, 1]
        self.coordinates[:, 1] = -all_data[1:, 0]

        snapshot = all_data[:, 2:]

        if isinstance(band_range, BandRangeBounds):
            wl_ids = [wl_id for wl_id, wl in enumerate(snapshot[0]) if band_range.lb < wl < band_range.rb]
        elif isinstance(band_range, BandRangeSet):
            wl_ids = [wl_id for wl_id, wl in enumerate(snapshot[0]) if wl in band_range.wls]
        else:
            raise Exception(f"Undefined class {band_range.__class__.__name__} passed")

        assert len(wl_ids) > 0
        self.wave_lengths = snapshot[0, wl_ids]

        # filter wavelengths
        band_data = snapshot[1:, wl_ids]
        # filter pixels
        # band_data = self.get_filtered_interval_pixels(band_data, mode='crop_right', part=0.2)

        self.mean_in_ch_by_px = band_data.mean(axis=0)
        self.left_dev_in_chs_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='left')
        self.right_dev_in_chs_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='right')

        self.mean_in_pxs_by_ch = band_data.mean(axis=1)
        self.left_dev_in_pxs_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='left')
        self.right_dev_in_pxs_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='right')

        self.band_data = band_data


class SnapshotMeta:
    def __init__(self, name: str, all_data: np.ndarray):
        self.name = name
        self.bands: Dict[str, BandData] = {
            band_name: BandData(band_range=band_range, all_data=all_data)
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


def get_features_df(group_features: Dict[str, List[SnapshotMeta]]) -> pd.DataFrame:
    band_metrics = [
        'mean_agg_in_channels',
        'dev_agg_in_channels',

        'mean_agg_in_pixels',
        'dev_agg_in_pixels',

        'too_low_pxs_mean',
        'too_high_pxs_mean'
    ]

    features_dict = {
        **{f"{band_name}_{band_metric}": [] for band_name in BANDS_DICT.keys() for band_metric in band_metrics},
        'class': [],
        'class_generalized': [],
        'name': []
    }

    for col_i, (class_name, class_snapshots) in enumerate(group_features.items()):
        for row_i, snapshot_meta in enumerate(class_snapshots):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta

            for band_key, band_value in snapshot_meta.bands.items():
                # cast for ide
                band_value: BandData = band_value

                features_dict[f'{band_key}_mean_agg_in_channels'].append(band_value.mean_in_ch_by_px.mean())
                features_dict[f'{band_key}_dev_agg_in_channels'].append(band_value.mean_dev_in_ch)

                features_dict[f'{band_key}_mean_agg_in_pixels'].append(band_value.mean_in_pxs_by_ch.mean())
                features_dict[f'{band_key}_dev_agg_in_pixels'].append(band_value.mean_dev_in_px)

                features_dict[f'{band_key}_too_low_pxs_mean'].append(band_value.get_too_low_pxs().mean())
                features_dict[f'{band_key}_too_high_pxs_mean'].append(band_value.get_too_high_pxs().mean())

            features_dict['class'].append(class_name)
            features_dict['class_generalized'].append('phyto' if 'phyto' in class_name else 'health')

            features_dict['name'].append(snapshot_meta.name)

    features_df = pd.DataFrame(features_dict)

    # normalize features
    # for key in features_df.keys():
    #     if key != 'class':
    #         features_df[key] /= features_df[key].max()

    return features_df


class Feature:
    def __init__(self, x_key: str, y_key: str, x_title: str, y_title: str, title: str):
        self.x_key = x_key
        self.y_key = y_key
        self.x_title = x_title
        self.y_title = y_title
        self.title = title


def clf_visualize(clf, x_keys: List[str], class_labels: List[str]):
    if isinstance(clf, DecisionTreeClassifier):
        fig = plt.figure(figsize=(25, 20))
        tree.plot_tree(clf, feature_names=x_keys, class_names=class_labels, filled=True)
        fig.savefig("decision_tree.png")
    elif isinstance(clf, LogisticRegression):
        pass  # TODO implement


def clf_build(features_df: pd.DataFrame, x_keys: List[str], y_key: str, method_name='lr') -> Dict[str, float]:
    x_all = features_df[x_keys]
    y_all = features_df[y_key]

    scaler = preprocessing.StandardScaler().fit(x_all)

    x_train, x_test, y_train, y_test = train_test_split(scaler.transform(x_all), y_all, test_size=0.33, random_state=42)

    if method_name == 'lr':
        clf = LogisticRegression(random_state=16)
    elif method_name == 'dt':
        clf = DecisionTreeClassifier(random_state=16)
    else:
        raise Exception(f"Undefined clf method = {method_name}")

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    str_to_int = lambda arr: [1 if el == 'health' else 2 for el in arr]

    return {
        'all': accuracy_score(y_test, clf.predict(x_test)).__round__(2),

        'train_confusion': confusion_matrix(y_pred=y_train_pred, y_true=y_train, labels=["health", "phyto"]),
        'train_auc': metrics.auc(
            *metrics.roc_curve(y_score=str_to_int(y_train_pred), y_true=str_to_int(y_train), pos_label=2)[:2]
        ).__round__(2),

        'test_confusion': confusion_matrix(y_pred=y_test_pred, y_true=y_test, labels=["health", "phyto"]),
        'test_auc': metrics.auc(
            *metrics.roc_curve(y_score=str_to_int(y_test_pred), y_true=str_to_int(y_test), pos_label=2)[:2]
        ).__round__(2),

        'cross_val': cross_val_score(clf, scaler.transform(x_all), y_all, cv=5).round(2),
        'clf': clf
    }


def draw_snapshots_in_features_space(features_df: pd.DataFrame):
    colors = ['#4FD51D', '#FF9999', '#E70000', "#830000", "#180000"]

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
        )
    ]

    for band_name in BANDS_DICT.keys():
        titles_band = [f"{feature.title} in {band_name} with logreg accuracy = " \
                       f"{clf_build(features_df=features_df, x_keys=[f'{band_name}_{feature.x_key}', f'{band_name}_{feature.y_key}'], y_key='class_generalized')['all']}"
                       for feature in features_list]

        fig = make_subplots(rows=1, cols=len(features_list), subplot_titles=titles_band)

        for class_name, color in zip(list(features_df['class'].unique()), colors):
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
        fig.write_html(f"{RES_DIR}/{band_name}/features_space_in_band={band_name}.html")


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
                             f"mean_dev={snapshot.bands[bname].mean_dev_in_px.round(2)}]"
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


def draw_files(classes_dict: Dict[str, List[str]], max_files_in_dir: int):
    classes_features_dict = parse_classes(classes_dict=classes_dict, max_files_in_dir=max_files_in_dir)

    for band_name in BANDS_DICT.keys():
        Path(f"{RES_DIR}/{band_name}").mkdir(exist_ok=True, parents=True)

    # draw hyperspectral glasses for every band
    for band_name in BANDS_DICT.keys():
        for i in range(2):
            step = 2
            i_st = i * step
            i_fin = i * step + step
            draw_hp_glasses(
                all_classes=[classes_features_dict[key][i_st:i_fin] for key in classes_features_dict.keys()],
                classes_names=[key for key in classes_features_dict.keys()],
                bname=band_name,
                res_path=f'{RES_DIR}/{band_name}/hp_glasses_for_snapshots[{i_st},{i_fin}].html'
            )

    # draw channels and pixels curves
    draw_snapshots_as_reflectance(classes_features_dict,
                                  res_path=f'{RES_DIR}/class_comparison_by_agg_in_channels.html',
                                  x_range=(0, 900), y_range=(0, 10_000), mode='ch')
    draw_snapshots_as_reflectance(classes_features_dict,
                                  res_path=f'{RES_DIR}/classes_comparison_by_agg_in_pixels.html',
                                  x_range=(0, 200), y_range=(8_000, 10_000), mode='px')

    # draw snapshots in features space
    features_df = get_features_df(group_features=classes_features_dict)
    draw_snapshots_in_features_space(features_df=features_df)

    return features_df


if __name__ == '__main__':
    features_df = draw_files(
        classes_dict={
            'health': [
                'csv/control/gala-control-bp-1_000',
                'csv/control/gala-control-bp-2_000',
                'csv/control/gala-control-bp-3_000',
                'csv/control/gala-control-bp-4_000',
            ],
            'phyto1': [
                'csv/phytophthora/gala-phytophthora-bp-1_000',
                'csv/phytophthora/gala-phytophthora-bp-5-1_000',
                # 'csv/phytophthora-ps-2_2-5_2/gala-phytophthora-2_2-5_2-1_000'
            ],
            'phyto2': [
                'csv/phytophthora/gala-phytophthora-bp-2_000',
                'csv/phytophthora/gala-phytophthora-bp-6-2_000',
                # 'csv/phytophthora-ps-2_2-5_2/gala-phytophthora-2_2-5_2-2_000'
            ],
            # 'phyto3': [
            #     'csv/phytophthora/gala-phytophthora-bp-3_000',
            #     'csv/phytophthora/gala-phytophthora-bp-7-3_000',
            # ],
            # 'phyto4': [
            #     'csv/phytophthora/gala-phytophthora-bp-4_000',
            #     'csv/phytophthora/gala-phytophthora-bp-8-4_000',
            # ]

        },
        max_files_in_dir=30
    )

    features_corr = features_df[[key for key in features_df.keys() if '_mean_agg_in_pixels' in key]].corr()
    features_by_classes = {class_name: features_df[features_df['class'] == class_name] for class_name in
                           features_df['class'].unique()}

    features_corr_by_classes = {
        class_name: class_df[[key for key in class_df.keys() if '_mean_agg_in_pixels' in key]].corr()
        for class_name, class_df in features_by_classes.items()
    }

    # clf_results = clf_build(
    #     features_df=features_df,
    #     x_keys=[
    #         'blue_mean_agg_in_pixels',
    #         'blue_dev_agg_in_pixels',
    #         'blue_too_low_pxs_mean',
    #         'blue_too_high_pxs_mean',
    #         'infrared_mean_agg_in_pixels',
    #         'infrared_dev_agg_in_pixels',
    #         'infrared_too_low_pxs_mean',
    #         'infrared_too_high_pxs_mean'
    #     ],
    #     y_key='class_generalized',
    #     method_name='lr'
    # )
    #
    # clf_visualize(
    #     clf=clf_results['clf'],
    #     x_keys=[
    #         'blue_mean_agg_in_pixels',
    #         'blue_dev_agg_in_pixels',
    #         'blue_too_low_pxs_mean',
    #         'blue_too_high_pxs_mean',
    #         'infrared_mean_agg_in_pixels',
    #         'infrared_dev_agg_in_pixels',
    #         'infrared_too_low_pxs_mean',
    #         'infrared_too_high_pxs_mean'
    #     ],
    #     class_labels=['health', 'phyto']
    # )
