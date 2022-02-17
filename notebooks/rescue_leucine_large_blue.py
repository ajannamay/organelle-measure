# This file tries to rescue the data of the leucine large experiment.
# Because the blue spectra image use a bad scanning wavelength range.
# The final hope lies in directly using Ilastik to segment both organelles.
# So before that we need some preprocessing.

import h5py
import numpy as np
from pathlib import Path
from skimage import io,exposure
from organelle_measure.tools import load_nd2_plane

folder_raw = "../data/raw/EYrainbow_leucine_large"
folder_out = "../data/leucine-large-blue"
for path_raw in Path(folder_raw).glob("spectral-blue*.nd2"):
    img_raw = load_nd2_plane(str(path_raw),frame="czyx",axes="t",idx=0).astype(int)
    # img_nor = exposure.rescale_intensity(img_raw,out_range=(0.,1.))
    with h5py.File(str(Path(folder_out)/f"{path_raw.stem}.hdf5"),"w") as f:
        f.create_dataset("data",data=img_raw,dtype="int16")
    
