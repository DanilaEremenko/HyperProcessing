import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib import pyplot as plt


def parse_tiff(img_path: Path, res_dir: Path, ref_threshold: Optional[float] = None, threshold_mode: str = 'left'):
    assert img_path.exists(), str(img_path)

    # if res_dir.exists():
    #     shutil.rmtree(res_dir)

    res_dir.mkdir(exist_ok=True, parents=True)

    image = tiff.imread(str(img_path))
    gray = image.mean(axis=2)
    gray /= gray.max()
    gray *= 254.
    gray = np.array(gray, dtype='uint8')
    plt.imshow(image.mean(axis=-1))
    plt.savefig(f"{res_dir}.png")

    original = image.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if threshold_mode == 'left':
        roi_pt_filter = lambda px, py: ROI[px, py].mean() < ref_threshold
        roi_agg_mask_fn = lambda ROI_agg: np.where(ROI_agg < ref_threshold, 1, 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    elif threshold_mode == 'right':
        roi_pt_filter = lambda px, py: ROI[px, py].mean() > ref_threshold
        roi_agg_mask_fn = lambda ROI_agg: np.where(ROI_agg > ref_threshold, 1, 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        raise Exception("Undefined roi filter")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, obtain bounding box coordinates, and extract ROI
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    WLS = np.arange(450, 871, 4)
    if ref_threshold is None:
        if threshold_mode == 'left':
            ref_threshold = original.max()
        elif threshold_mode == 'right':
            ref_threshold = original.min()
        else:
            raise Exception(f'Undefined threshold mode = {threshold_mode}')

    for ci, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        ROI = original[y:y + h, x:x + w]
        curr_df = pd.DataFrame(
            [
                {1.: px, 2.: py, **{WLS[ch_i]: pv for ch_i, pv in enumerate(ROI[px, py])}}
                for px in range(ROI.shape[0])
                for py in range(ROI.shape[1])
                if roi_pt_filter(px, py)
            ]
        )
        snap_name = f"{img_path.name.replace('cube.tiff', '')}view__{ci}"
        curr_df.to_csv(res_dir.joinpath(f"{snap_name}.csv"), index=False)

        ROI_agg = ROI.mean(axis=-1)
        plt.imshow(ROI_agg)
        plt.savefig(res_dir.joinpath(f"{snap_name}_source.png"))

        plt.imshow(roi_agg_mask_fn(ROI_agg))
        plt.savefig(res_dir.joinpath(f"{snap_name}_filtered.png"))

        # leaf_pixels = np.where(ROI_agg < hyper_thresh, 0, 1)
        # plt.xlim((ROI_agg.shape[1]), 0)
        # plt.ylim((ROI_agg.shape[0]), 0)
        # plt.imshow(leaf_pixels)
        # plt.show()
        #
        # background_pixels = np.where(ROI_agg >= hyper_thresh, 0, 1)
        # plt.imshow(background_pixels)
        # plt.xlim((ROI_agg.shape[1]), 0)
        # plt.ylim((ROI_agg.shape[0]), 0)
        # plt.show()

        # cv2.imshow('image', image)
        # cv2.imshow('thresh', thresh)
        # cv2.imshow('dilate', dilate)
        # cv2.waitKey()


if __name__ == '__main__':
    # parse_tiff(
    #     img_path=Path('cubert_data/puccinia/wheat-puccinia-0_000_000_snapshot_cube.tiff'),
    #     res_dir=Path('csv/new_data/old_snap_example'),
    #     ref_threshold=9_500,
    #     threshold_mode='left'
    # )
    #
    # parse_tiff(
    #     img_path=Path('Cubert2/2023_08_12/cochle-control4_000/export/cochle-control4_000_000_snapshot_REF.tiff'),
    #     res_dir=Path('csv/new_data/new_snap_example'),
    #     ref_threshold=1_600,
    #     threshold_mode='right'
    # )

    parse_dir = Path(f'../../datasets/2024_05_17')
    img_files = [
        Path(f"{dir}/{file}")
        for group_name in ['contr', 'exp']
        for dir, subdirs, files in os.walk(parse_dir)
        for file in files if 'tiff' in file and 'REF' in file
    ]

    for i, img_file in enumerate(img_files):
        print(f"parse snapshot {i}/{len(img_files)}")
        parse_tiff(
            img_path=img_file,
            res_dir=Path(f'csv/{parse_dir.name}').joinpath(img_file.parent.parent.name),
            # ref_threshold=1600,
            ref_threshold=1300,
            threshold_mode='left',
            # ref_threshold=1200,
            # threshold_mode='left'
        )
