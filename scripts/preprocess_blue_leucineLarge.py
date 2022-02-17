import h5py
import numpy as np
from pathlib import Path
from skimage import filters
from organelle_measure.tools import load_nd2_plane

folder_raw = "../data/raw/EYrainbow_leucine_large"
folder_out = "../data/leucine-large-blue-gaussian"
for path_raw in Path(folder_raw).glob("spectral-blue*.nd2"):
    img_raw = load_nd2_plane(str(path_raw),frame="czyx",axes="t",idx=0).astype(int)
    img_gau = np.zeros_like(img_raw,dtype=int)
    for c in range(len(img_raw)):
        img_gau[c] = filters.gaussian(img_raw[c],sigma=0.75,preserve_range=True)
    # img_nor = exposure.rescale_intensity(img_raw,out_range=(0.,1.))
    with h5py.File(str(Path(folder_out)/f"{path_raw.stem}.hdf5"),"w") as f:
        f.create_dataset("data",data=img_gau,dtype="uint16")