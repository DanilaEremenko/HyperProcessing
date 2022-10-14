from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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

    def get_clusters_features(self, X) -> Tuple[float, float, float]:
        # X = X[:, :9]
        X_normalized = np.zeros(shape=X.shape)
        for ch in range(X.shape[1]):
            X_curr = X[:, ch]
            X_normalized[:, ch] = (X_curr - X_curr.mean()) / (X_curr.max() - X_curr.min())

        c_num = 3
        kmeans = KMeans(n_clusters=c_num, random_state=42).fit(X_normalized)
        labels = kmeans.labels_
        clusters = [X[[i for i, pt_label in enumerate(labels) if pt_label == filt_label], :-2]
                    for filt_label in range(c_num)]

        k_low = clusters[np.argmin([cl.mean() for cl in clusters])]
        k_high = clusters[np.argmax([cl.mean() for cl in clusters])]

        k_ratio = min([len(k_low), len(k_high)]) / max([len(k_low), len(k_high)])
        k_het = abs(k_low.mean() - k_high.mean()) / (k_low.mean() + k_high.mean()) if k_ratio > 0.25 \
            else 0

        k_low_part = len(k_low) / len(X)
        k_high_part = len(k_high) / len(X)

        return k_het, k_low_part, k_high_part

    @property
    def mean_dev_in_px(self) -> np.float_:
        return self.right_dev_in_pxs_by_ch.mean() - self.left_dev_in_pxs_by_ch.mean()

    @property
    def mean_dev_in_ch(self) -> np.float_:
        return self.right_dev_in_chs_by_px.mean() - self.left_dev_in_chs_by_px.mean()

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

        self.padded_data = self.get_padded_data(
            coordinates=self.coordinates,
            band_data=self.band_data
        )

        self.k_het, self.k_low_part, self.k_high_part \
            = self.get_clusters_features(X=np.concatenate((self.band_data, self.coordinates), axis=1))


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


class SnapshotMeta:
    def __init__(self, name: str, all_data: np.ndarray):
        self.name = name
        self.bands: Dict[str, BandData] = {
            band_name: BandData(band_range=band_range, all_data=all_data)
            for band_name, band_range in BANDS_DICT.items()
        }
