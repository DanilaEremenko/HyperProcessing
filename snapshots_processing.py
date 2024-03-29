import os
from math import sin, sqrt
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops


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
            # raise Exception("All channels have same value for pixel")
            return dev_arr
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


class ClusterData:
    def __init__(self, cl_data: np.ndarray, src_size: int):
        self.part = len(cl_data) / src_size
        self.het = self.part * cl_data.std() / cl_data.mean()
        self.mean = cl_data.mean()


class ClusterFeatures:
    def __init__(self, c_low: ClusterData, c_high: ClusterData):
        self.c_low = c_low
        self.c_high = c_high
        self.all_het = 100 * abs(c_low.part - c_high.part) * abs(c_low.mean - c_high.mean) / (c_low.mean + c_high.mean)

    def __str__(self):
        return f"c_low_part = {self.c_low.part.__round__(2)}, " \
               f"c_low_mean = {self.c_low.mean.__round__(2)}, " \
               f"c_low_het = {self.c_low.het.__round__(2)}, " \
               f"c_high_mean = {self.c_high.mean.__round__(2)}, " \
               f"c_high_het = {self.c_high.het.__round__(2)}"


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
        # ch_id = [i for i, wl in enumerate(self.wave_lengths) if wl == target_wl][0]
        target_wl = max(450., target_wl)
        target_wl = min(870., target_wl)
        ch_id = int((target_wl - 450.) / 4)
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

    def get_clusters_features(self, X) -> ClusterFeatures:
        # X = X[:, :9]
        X_normalized = np.zeros(shape=X.shape)
        for ch in range(X.shape[1]):
            X_curr = X[:, ch]
            X_normalized[:, ch] = (X_curr - X_curr.mean()) / (X_curr.max() - X_curr.min())

        c_num = 2
        kmeans = KMeans(n_clusters=c_num, random_state=42).fit(X_normalized)
        labels = kmeans.labels_
        clusters = [X[[i for i, pt_label in enumerate(labels) if pt_label == filt_label], :-2]
                    for filt_label in range(c_num)]

        # if k_smallest_size > 0.25 \
        # else 0

        return ClusterFeatures(
            c_low=ClusterData(cl_data=clusters[np.argmin([cl.mean() for cl in clusters])], src_size=len(X)),
            c_high=ClusterData(cl_data=clusters[np.argmax([cl.mean() for cl in clusters])], src_size=len(X))
        )

    @property
    def mean_dev_in_px(self) -> np.float_:
        # return self.right_dev_in_pxs_by_ch.mean() - self.left_dev_in_pxs_by_ch.mean()
        return self.band_data.mean(axis=1).std()

    @property
    def mean_dev_in_ch(self) -> np.float_:
        # return self.right_dev_in_chs_by_px.mean() - self.left_dev_in_chs_by_px.mean()
        return self.band_data.mean(axis=0).std()

    @staticmethod
    def get_padded_data(coordinates: np.ndarray, band_data: np.ndarray) -> np.ndarray:
        coordinates_norm = coordinates.copy()
        coordinates_norm[:, 0] -= coordinates_norm[:, 0].min()
        coordinates_norm[:, 1] -= coordinates_norm[:, 1].min()

        coordinates_norm = np.array(coordinates_norm, dtype='uint8')

        y_size = coordinates_norm[:, 0].max()
        x_size = coordinates_norm[:, 1].max()

        padded_data = np.zeros(shape=(y_size, x_size, band_data.shape[-1]))

        for (y, x), band_val in zip(coordinates_norm, band_data):
            padded_data[y - 1, x - 1] = band_val

        return padded_data

    @property
    def padded_data(self) -> np.array:
        if self._padded_data is None:
            self._padded_data = self.get_padded_data(
                coordinates=self.coordinates,
                band_data=self.band_data
            )

        return self._padded_data

    @property
    def cl_features(self):
        if self._cl_features is None:
            self._cl_features = self.get_clusters_features(X=np.concatenate((self.band_data, self.coordinates), axis=1))
        return self._cl_features

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
        # self.left_dev_in_chs_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='left')
        # self.right_dev_in_chs_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='right')

        self.mean_in_pxs_by_ch = band_data.mean(axis=1)
        # self.left_dev_in_pxs_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='left')
        # self.right_dev_in_pxs_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='right')

        self.band_data = band_data

        # let's go lazy
        self._padded_data = None
        self._cl_features = None


class BandYri:
    def get_band_data_in_ch_id(self, ch_id: int) -> np.ndarray:
        return self.band_data[:, 0]

    def __init__(self, all_data):
        # separate coordinates and snapshot data
        self.coordinates = all_data[1:, :2].copy()

        # coordinates matching with image format
        self.coordinates[:, 0] = all_data[1:, 1]
        self.coordinates[:, 1] = -all_data[1:, 0]

        self.wave_lengths = [0]

        # filter wavelengths
        band_data = (all_data[:, 72] - all_data[:, 2]) / (all_data[:, 72] + all_data[:, 2]) \
                    + 0.5 * all_data[:, 73]
        band_data = band_data[1:]
        band_data = np.expand_dims(band_data, axis=1)

        # filter pixels
        # band_data = self.get_filtered_interval_pixels(band_data, mode='crop_right', part=0.2)

        self.mean_in_ch_by_px = band_data.mean(axis=0)
        # self.left_dev_in_chs_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='left')
        # self.right_dev_in_chs_by_px = band_data.mean(axis=0) + calculate_dev_in_channels(band_data, mode='right')

        self.mean_in_pxs_by_ch = band_data.mean(axis=1)
        # self.left_dev_in_pxs_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='left')
        # self.right_dev_in_pxs_by_ch = band_data.mean(axis=1) + calculate_dev_in_pixels(band_data, mode='right')

        self.band_data = band_data

        # let's go lazy
        self._padded_data = None
        self._cl_features = None


WLS = np.arange(450, 871, 4)

BANDS_DICT = {
    # **{
    #     f'{wl1}-{wl2}': BandRangeSet(wls=[wl1, wl2])
    #     for i, (wl1, wl2) in enumerate(zip(WLS, WLS[1:]))
    #     if i % 2 == 0
    # },
    **{
        f'{wl}': BandRangeSet(wls=[wl])
        for i, wl in enumerate(WLS)
    },

    # 'blue_set': BandRangeSet(wls=[450]),
    # 'blue': BandRangeBounds(440, 500),
    # 'green': BandRangeBounds(500, 590),
    # 'red': BandRangeBounds(590, 780),
    # 'visible': BandRangeBounds(440, 780),
    # 'infrared': BandRangeBounds(780, 1000),

    # 'blue': BandRangeBounds(440, 485),
    # 'cyan': BandRangeBounds(485, 500),
    # 'green': BandRangeBounds(500, 565),
    # 'yellow': BandRangeBounds(565, 590),
    # 'orange': BandRangeBounds(590, 625),
    # 'red': BandRangeBounds(625, 780),
    # 'visible': BandRangeBounds(440, 780),
    # 'infrared': BandRangeBounds(780, 1000),

    'all': BandRangeBounds(400, 1000),
}


class SnapshotMeta:
    def __init__(self, dir_path: str, name: str, all_data: np.ndarray):
        self.dir = dir_path
        self.name = name
        self.bands: Dict[str, BandData] = {
            band_name: BandData(band_range=band_range, all_data=all_data)
            for band_name, band_range in BANDS_DICT.items()
        }
        # self.bands['YRI'] = BandYri(all_data=all_data)

    def get_tf_gr_features_for_band(self, wl: str) -> dict:
        image = self.bands[wl].padded_data[:, :, 0]
        bins = np.linspace(4000, 10_000, 16)
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        coefs = [0, 1 / 4, 1 / 2, 3 / 4]
        angles_radians = [np.pi * coef for coef in coefs]
        angles_degree = [180 * coef for coef in coefs]

        matrix_coocurrence = graycomatrix(
            image=inds,
            distances=[1],
            angles=angles_radians,
            levels=max_value,
            normed=False,
            symmetric=False
        )

        # GLCM properties
        def features_by_angles(feature_name: str) -> dict:
            # features_vect = graycoprops(matrix_coocurrence, feature_name).flatten()
            # return {f"{wl}_tf_gr_{feature_name}_{angle}": tf for tf, angle in zip(features_vect, angles_degree)}
            return {f"{wl}_tf_gr_{feature_name}": graycoprops(matrix_coocurrence, feature_name).mean()}

        feature_keys = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

        feature_list = [features_by_angles(feature_name=feature_key) for feature_key in feature_keys]
        feature_dict = {key: value for d in feature_list for key, value in d.items()}

        return feature_dict

    def get_sep_tf_features(self) -> dict:
        feature_list = [self.get_tf_gr_features_for_band(wl=str(wl)) for wl in WLS]
        feature_dict = {key: value for d in feature_list for key, value in d.items()}
        return feature_dict

    def get_indexes_features(self) -> dict:

        R_HASH = {}

        def R(wl) -> float:
            if wl not in R_HASH.keys():
                R_HASH[wl] = self.bands['all'].get_band_data_in_wl(target_wl=wl).mean()

            return R_HASH[wl]

        return {
            # 'ARI': 1 / R(550) - 1 / R(702),  # green right & red middle
            # 'BGI': R(450) / R(550),  # blue left & green right
            # 'BRI': R(450) / R(690),  # blue left & red middle
            # 'CAI': R(450) / R(690),  # blue left & red middle
            #
            # 'CRI1': 1 / R(510) - 1 / R(550),  # green left & green right
            # 'CRI2': 1 / R(510) - 1 / R(702),  # green left & red middle
            #
            # 'CSI1': R(694) / R(450),  # red middle & blue left
            # 'CSI2': R(694) / R(762),  # red middle & red right
            #
            # 'CUR': R(674) * R(550) / R(682) ** 2,  # red middle & green right
            # 'gNDVI': (R(750) - R(550)) / (R(750) + R(550)),  # red right & green right
            # 'hNDVI': (R(826) - R(666)) / (R(826) + R(666)),  # infr left & red left
            # 'NPCI': (R(682) - R(450)) / R(682) + R(450),

            # TOPIC STUFF
            'CRI1': 1 / R(510) - 1 / R(550),  # green left & green right
            'CRI2': 1 / R(510) - 1 / R(702),  # green left & red middle
            'LCI': (R(850) - R(710)) / (R(850) + R(682)),

            'PSSRa': R(802) / R(678),
            'PSSRb': R(802) / R(650),
            'PSSRc': R(802) / R(500),

            'SR(Chla)': R(674) / R(702),
            'SR(Chlb)': R(674) / (R(650) * R(702)),
            'SR(Chlb2)': R(670) / R(710),
            'SR(Chltot)': R(762) / R(502),

            # FOR CLASSIFICATION FROM PAPERS
            'PRI': (R(570) - R(534)) / (R(570) + R(534)),
            'PHRI': (R(550) - R(534)) / (R(570) + R(534)),
            'NDVI': (R(830) - R(674)) / (R(830) + R(674)),
            'MSR': (R(800) / R(670) - 1) / sqrt(R(800) / R(670) + 1),
            'TVI': 0.5 * ((120 * R(750) - R(550)) - 200 * (R(670) - R(550))),
            'SIPI': (R(800) - R(445)) / (R(800) - R(680)),
            'NPCI': (R(680) - R(430)) / (R(680) + R(430)),
            'ARI': R(550) ** (-1) - R(700) ** (-1),
            'GI': R(554) / R(667),
            'TCARI': 3 * ((R(700) - R(675)) - 0.2 * ((R(700) - R(500)) / (R(700) - R(670)))),
            'PSRI': (R(680) - R(500)) / R(750),
            'RVSI': (R(712) + R(752)) / 2 - R(732),
            'NRI': (R(570) - R(670)) / (R(570) + R(670)),
            'YRI': ((R(730) - R(419)) / (R(730) + R(419))) + 0.5 * R(736)
        }

    def get_sep_band_features(self):
        features_dict = {'dir': self.dir, 'name': self.name}

        for band_name, band_data in self.bands.items():
            # cast for ide
            band_data: BandData = band_data

            curr_band_features = {
                'all_pixels_mean': band_data.mean_in_pxs_by_ch.mean(),
                # 'all_pixels_std': band_data.mean_dev_in_px,

                # 'min_max_diff': (band_data.mean_in_pxs_by_ch.max() - band_data.mean_in_pxs_by_ch.min()) /
                #                 (band_data.mean_in_pxs_by_ch.max() + band_data.mean_in_pxs_by_ch.min()),
                # 'dev_agg_in_channels': band_data.mean_dev_in_ch,

                # 'too_low_pxs_mean': band_data.get_too_low_pxs().mean(),
                # 'too_high_pxs_mean': band_data.get_too_high_pxs().mean(),

                # 'cl_all_het': band_data.cl_features.all_het,
                #
                # 'cl_low_mean': band_data.cl_features.c_low.mean,
                # 'cl_high_mean': band_data.cl_features.c_high.mean,
                #
                # 'cl_low_het': band_data.cl_features.c_low.het,
                # 'cl_high_het': band_data.cl_features.c_high.het,
                #
                # 'cl_low_part': band_data.cl_features.c_low.part,
                # 'cl_high_part': band_data.cl_features.c_high.part
            }

            generate_func_map = {
                # '^2': lambda px: px ** 2,
                # '^3': lambda px: px ** 3,
                # 'sin': lambda px: sin(px)
            }

            curr_band_features = {
                **curr_band_features,
                **{
                    f"{src_name}_{postf}": func(src_val)
                    for src_name, src_val in curr_band_features.items()
                    for postf, func in generate_func_map.items()
                }
            }

            curr_band_features = {f"{band_name}_{key}": val for key, val in curr_band_features.items()}

            features_dict = {**features_dict, **curr_band_features}

        return features_dict

    def get_features_dict(self) -> dict:
        return {
            **self.get_sep_band_features(),
            **self.get_sep_tf_features(),
            **self.get_indexes_features()
        }


def parse_classes(classes_dict: Dict[str, List[str]], max_files_in_dir: int) -> Dict[str, List[SnapshotMeta]]:
    classes_features: Dict[str, List[SnapshotMeta]] = {class_name: [] for class_name in classes_dict.keys()}
    for class_name, class_dirs in classes_dict.items():
        features = classes_features[class_name]
        for dir_path in class_dirs:
            print(f"parsing snapshots from {dir_path}")
            files = list(os.listdir(dir_path))[:max_files_in_dir]
            files = [file for file in files if 'csv' in file]
            assert len(files) >= 1
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
