from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from osgeo import gdal


class SnapPushBroom:
    def __init__(self, path: str):
        # gdal.GetDriverByName('EHdr').Register()

        self.img: gdal.Dataset = gdal.Open(path)
        self.units = self.img.GetMetadata()['wavelength_units']
        self.wavelengths = [float(val.replace(' nanometers', ''))
                            for key, val in self.img.GetMetadata().items() if 'Band' in key]

    def corr_in_bands(self, b1: int, b2: int):
        assert b1 < len(self.wavelengths)
        assert b2 < len(self.wavelengths)

        return np.corrcoef(
            self.img.GetRasterBand(b1).ReadAsArray().flatten(),
            self.img.GetRasterBand(b2).ReadAsArray().flatten()
        )[0][1]

    def draw_bands(self, ch_from: int, ch_to: int, snap_per_row: int):
        ch_num = ch_to - ch_from
        assert ch_num > 0

        assert ch_num > snap_per_row

        nrows = ch_num // (snap_per_row + 1) + 1
        ncols = min(snap_per_row, ch_num)

        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 2.5, nrows * 2.5))

        for i, wl in enumerate(self.wavelengths[ch_from:ch_to]):
            band = self.img.GetRasterBand(i + 1)
            data = band.ReadAsArray()

            row_i = i // snap_per_row
            col_i = i % snap_per_row

            axes[row_i, col_i].imshow(data)
            axes[row_i, col_i].set_title(f"wl = {wl}")

        fig.tight_layout()
        fig.show()


if __name__ == '__main__':
    pref_path = Path('../GIPER_AFI/20220713/20220713_Тестоввые площадки№1 и №2_80 метров/')

    files = [
        'Testing_1-1/Testing_1_Pika_L_1-radiance.bil',
        'Testing_1-2/Testing_1_Pika_L_2-radiance.bil'
    ]

    files = [f"{pref_path}/{file}" for file in files]

    assert all([Path(file).exists() for file in files])

    snapshots = [SnapPushBroom(path=file) for file in files]

    for i in range(0, 120, 20):
        snapshots[0].draw_bands(i, i + 20, snap_per_row=5)
