from pathlib import Path
from typing import List, Dict

import numpy as np
from numpy import genfromtxt
import os
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

from clf import clf_build
from drawing import draw_hp_glasses, draw_snapshots_as_reflectance, draw_snapshots_in_features_space, \
    draw_snapshots_in_all_paired_features_space, draw_tsne_plotly, draw_tsne_matplot
from experiments import *
from snapshots_processing import SnapshotMeta, BandData, BANDS_DICT

RES_DIR = Path('sub-wheat_comparison_with_indexes_filtered_each_wl_imp_analyze_topic_stuf')
RES_DIR.mkdir(exist_ok=True)

CLASSES_DICT = {
    # **POTATO_OLD,
    # **POTATO_NEW,

    # **WHEAT_ALL,
    # **WHEAT_ALL_FILTERED,

    **WHEAT_ALL_CLEAR_EXP
}

MAX_FILES_IN_DIR = 100


def draw_klebs_np(ax, snapshot: np.ndarray, title: str):
    # for i in range(0, len(snapshot) - 1, 10):
    #     plt.plot(snapshot[0], snapshot[i + 1])
    ax.plot(snapshot[0], snapshot[1:].mean(axis=0))
    ax.set_xlim(400, 900)
    ax.set_ylim(0, 10_000)
    ax.set_title(title)


def parse_classes(classes_dict: Dict[str, List[str]], max_files_in_dir: int) -> Dict[str, List[SnapshotMeta]]:
    classes_features: Dict[str, List[SnapshotMeta]] = {class_name: [] for class_name in classes_dict.keys()}
    for class_name, class_dirs in classes_dict.items():
        features = classes_features[class_name]
        for dir_path in class_dirs:
            print(f"parsing snapshots from {dir_path}")
            files = list(os.listdir(dir_path))[:max_files_in_dir]
            for file_id, file in enumerate(files):
                print(f"  parsing file={file}")
                all_data = genfromtxt(
                    f'{dir_path}/{file}',
                    delimiter=','
                )

                if len(all_data) < 20:
                    print(f"  {file} small leaf, scip..")
                else:
                    features.append(
                        SnapshotMeta(dir_path=dir_path, name=file, all_data=all_data)
                    )

    return classes_features


def get_features_df(group_features: Dict[str, List[SnapshotMeta]]) -> pd.DataFrame:
    features_list = []

    for col_i, (class_name, class_snapshots) in enumerate(group_features.items()):
        for row_i, snapshot_meta in enumerate(class_snapshots):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta
            print(f'getting features for {snapshot_meta.name}')
            features_dict = snapshot_meta.get_features_dict()
            features_dict['class'] = class_name
            features_dict['class_generalized'] = 'phyto' if 'phyto' in class_name else 'health'

            features_list.append(features_dict)

    features_df = pd.DataFrame(features_list)

    # normalize features
    # for key in features_df.keys():
    #     if key != 'class':
    #         features_df[key] /= features_df[key].max()

    return features_df


def draw_files(classes_features_dict: Dict[str, List[SnapshotMeta]], features_df: pd.DataFrame):
    for band_name in BANDS_DICT.keys():
        Path(f"{RES_DIR}/{band_name}").mkdir(exist_ok=True, parents=True)

    # draw hyperspectral glasses for every band
    for band_name in BANDS_DICT.keys():
        for i in range(4):
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
    # draw_snapshots_as_reflectance(classes_features_dict,
    #                               res_path=f'{RES_DIR}/class_comparison_by_agg_in_channels.html',
    #                               x_range=(0, 900), y_range=(0, 10_000), mode='ch')
    # draw_snapshots_as_reflectance(classes_features_dict,
    #                               res_path=f'{RES_DIR}/classes_comparison_by_agg_in_pixels.html',
    #                               x_range=(0, 200), y_range=(8_000, 10_000), mode='px')

    # draw snapshots in features space
    # draw_snapshots_in_features_space(features_df=features_df, res_dir=RES_DIR)
    draw_snapshots_in_all_paired_features_space(features_df=features_df, res_dir=RES_DIR)


def main() -> pd.DataFrame:
    classes_features_dict = parse_classes(
        classes_dict=CLASSES_DICT,
        max_files_in_dir=MAX_FILES_IN_DIR
    )

    features_df = get_features_df(group_features=classes_features_dict)

    draw_files(classes_features_dict=classes_features_dict, features_df=features_df)

    return features_df


def get_statistics_grouped_by_key_df(features_df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    statistics_list = []
    for group_val in features_df[group_key].unique():
        class_df = features_df[features_df[group_key] == group_val]
        curr_statistics = [
            {
                'group': group_val,
                'feature': feature,
                'mean': class_df[feature].mean(),
                'std': class_df[feature].std()
            }
            for feature in class_df.keys() if is_numeric_dtype(features_df[feature])
        ]

        statistics_list.extend(curr_statistics)

    return pd.DataFrame(statistics_list)


if __name__ == '__main__':
    # features_df = main()
    #
    # features_df.to_csv(f"{RES_DIR}/features.csv", index=False)
    features_df = pd.read_csv(f"{RES_DIR}/features.csv")

    stat_df = get_statistics_grouped_by_key_df(features_df=features_df, group_key='class')
    # stat_df[stat_df['feature'] == 'hNDVI']

    features_corr = features_df[[key for key in features_df.keys()]].corr(numeric_only=True)

    features_by_classes = {class_name: features_df[features_df['class'] == class_name]
                           for class_name in features_df['class'].unique()}

    features_corr_by_classes = {
        class_name: class_df[[key for key in class_df.keys()]].corr(numeric_only=True)
        for class_name, class_df in features_by_classes.items()
    }

    clf_build(
        # fit_df=pd.concat([features_df.iloc[800:1000], features_df.iloc[1300:1500]]),
        # eval_df=pd.concat([features_df.iloc[1600:1800], features_df.iloc[2000:2200]]),
        fit_df=features_df.iloc[0:400],
        eval_df=features_df.iloc[400:800],
        # features_df=features_df.iloc[[i for i, name in enumerate(list(features_df['dir']))
        #                               if int(name[-1]) in [4, 5, 6, 7]]],
        # features_df.iloc[[i for i, name in enumerate(list(features_df['dir']))
        #                   if '000' == name[-3:] and 'day' not in name and int(name[-5]) in [4, 5, 6, 7]]],
        # features_df = features_df.iloc[[i for i, name in enumerate(list(features_df['dir']))
        #                                 if 'day' in name and int(name.split('day')[1][0]) in [4, 5, 6, 7]]],
        x_keys=[
            *[
                f"{_range}_{pred}"
                for pred in [
                    'all_pixels_mean',
                    # 'all_pixels_std',
                    # 'dev_agg_in_pixels',
                    # 'dev_agg_in_channels',  # good one for all bands
                    # 'too_low_pxs_mean', 'too_high_pxs_mean',
                    # 'cl_all_het', 'cl_low_het', 'cl_high_het', 'cl_high_part', 'cl_low_part',
                ]
                # for range in BANDS_DICT.keys()
                # for _range in [str(wl) for wl in range(514, 550, 4)]
                # for _range in [wl for wl in np.arange(450, 871, 4)]
                # for _range in [762, 650, 470, 766, 466, 706, 502, 718, 854, 722, 714]  # lr
                for _range in [502, 466, 598, 718, 534, 766, 694, 650, 866, 602, 858]  # svm

            ],
            *[
                # 'ARI', 'BGI', 'BRI', 'CRI1', 'CRI2', 'CSI1', 'CSI2', 'CUR', 'gNDVI', 'hNDVI',
                # 'NPCI'
            ]
        ],
        y_key='class_generalized',
        # method_name='lr',
        method_name='svc',
        # clf_args=dict(kernel='rbf', C=3.),
        # dec_analyze=True,
        # clf_pretrained=clf_results['clf'],
        scaler_fit_on_all=False
    )

    # draw_tsne_matplot(
    #     features_df=features_df,
    #     # features_df=features_df.iloc[[i for i, name in enumerate(list(features_df['dir']))
    #     #                               if int(name[-1]) in [4, 5, 6, 7]]],
    #     # features_df.iloc[[i for i, name in enumerate(list(features_df['dir']))
    #     #                   if '000' == name[-3:] and 'day' not in name and int(name[-5]) in [4, 5, 6, 7]]],
    #     # features_df = features_df.iloc[[i for i, name in enumerate(list(features_df['dir']))
    #     #                                 if 'day' in name and int(name.split('day')[1][0]) in [4, 5, 6, 7]]],
    #     features=[
    #         *[
    #             f"{_range}_{pred}"
    #             for pred in [
    #                 'all_pixels_mean',
    #                 # 'all_pixels_std',
    #                 # 'dev_agg_in_pixels',
    #                 # 'dev_agg_in_channels',  # good one for all bands
    #                 # 'too_low_pxs_mean', 'too_high_pxs_mean',
    #                 # 'cl_all_het', 'cl_low_het', 'cl_high_het', 'cl_high_part', 'cl_low_part',
    #             ]
    #             # for _range in [wl for wl in np.arange(450, 871, 4)]
    #             for _range in [762, 650, 470, 766, 466, 706, 502, 718, 854, 722, 714]
    #         ],
    #         *[
    #             # 'ARI', 'BGI', 'BRI', 'CRI1', 'CRI2', 'CSI1', 'CSI2', 'CUR', 'gNDVI', 'hNDVI',
    #             # 'NPCI'
    #         ]
    #     ],
    #     save_pref='tsne_important'
    # )
