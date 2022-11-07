from pathlib import Path
from typing import List, Dict

import numpy as np
from numpy import genfromtxt
import os
import pandas as pd

from clf import clf_build
from drawing import draw_hp_glasses, draw_snapshots_as_reflectance, draw_snapshots_in_features_space, \
    draw_snapshots_in_all_paired_features_space, draw_tsne
from snapshots_processing import SnapshotMeta, BandData, BANDS_DICT

RES_DIR = Path('comparison_no_filt_tryy')
RES_DIR.mkdir(exist_ok=True)


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
                all_data = genfromtxt(
                    f'{dir_path}/{file}',
                    delimiter=','
                )
                features.append(
                    SnapshotMeta(name=file, all_data=all_data)
                )
    return classes_features


def get_features_df(group_features: Dict[str, List[SnapshotMeta]]) -> pd.DataFrame:
    features_list = []

    for col_i, (class_name, class_snapshots) in enumerate(group_features.items()):
        for row_i, snapshot_meta in enumerate(class_snapshots):
            # cast for ide
            snapshot_meta: SnapshotMeta = snapshot_meta

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
    draw_snapshots_as_reflectance(classes_features_dict,
                                  res_path=f'{RES_DIR}/class_comparison_by_agg_in_channels.html',
                                  x_range=(0, 900), y_range=(0, 10_000), mode='ch')
    draw_snapshots_as_reflectance(classes_features_dict,
                                  res_path=f'{RES_DIR}/classes_comparison_by_agg_in_pixels.html',
                                  x_range=(0, 200), y_range=(8_000, 10_000), mode='px')

    # draw snapshots in features space
    draw_snapshots_in_features_space(features_df=features_df, res_dir=RES_DIR)
    draw_snapshots_in_all_paired_features_space(features_df=features_df, res_dir=RES_DIR)


def main() -> pd.DataFrame:
    classes_features_dict = parse_classes(
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

    features_df = get_features_df(group_features=classes_features_dict)

    draw_files(classes_features_dict=classes_features_dict, features_df=features_df)

    return features_df


if __name__ == '__main__':
    features_df = main()

    features_corr = features_df[[key for key in features_df.keys()]].corr()

    features_by_classes = {class_name: features_df[features_df['class'] == class_name]
                           for class_name in features_df['class'].unique()}

    features_corr_by_classes = {
        class_name: class_df[[key for key in class_df.keys()]].corr()
        for class_name, class_df in features_by_classes.items()
    }

    clf_results = clf_build(
        features_df=features_df,
        x_keys=[
            f"{range}_{pred}"
            for pred in ['mean_agg_in_pixels', 'dev_agg_in_pixels', 'dev_agg_in_channels',
                         'too_low_pxs_mean', 'too_high_pxs_mean',
                         'cl_all_het', 'cl_low_het', 'cl_high_het', 'cl_high_part', 'cl_low_part']
            for range in ['blue', 'cyan', 'infrared', 'green']
        ],
        y_key='class_generalized',
        method_name='lr',
        # method_name='svc',
        # clf_args=dict(kernel='rbf', C=3.),
        # dec_analyze=True
    )

    draw_tsne(
        features_df=features_df,
        features=[
            f"{range}_{pred}"
            for pred in ['mean_agg_in_pixels', 'dev_agg_in_pixels', 'too_low_pxs_mean', 'too_high_pxs_mean',
                         'cl_low_het', 'cl_high_het', 'cl_high_part', 'cl_low_part']
            for range in ['blue', 'infrared', 'green']
        ],
    )
