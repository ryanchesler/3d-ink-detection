import cv2
from tqdm import tqdm

import pathlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pathlib
import h5py
import monai
final_coords = []
with h5py.File('/data/all_labels_train.hdf5', 'r') as f:
    crop_size = 128
    image = f["scan_counts"]
    macro_generator = monai.data.utils.iter_patch_position(image.shape, (crop_size, crop_size, crop_size), overlap=0, padded=False)
    for coords in tqdm(macro_generator):
        counts = f["scan_counts"][coords[0]:(coords[0]+((crop_size))),
                            coords[1]:(coords[1]+((crop_size))),
                            coords[2]:(coords[2]+((crop_size)))]

        if counts.max() > 0:
            counts[counts == 0] = 1
            volume = f["scan_predictions"][coords[0]:(coords[0]+((crop_size))),
                            coords[1]:(coords[1]+((crop_size))),
                            coords[2]:(coords[2]+((crop_size)))]
            final_label = (volume/counts)
            if (final_label > (255*.8)).mean() > 0:
                print("hit")
            final_coords.append([coords[0], coords[1], coords[2]])
coords = np.array(final_coords)
print(coords.shape)
np.save("train_coords_128.npy", coords)
        