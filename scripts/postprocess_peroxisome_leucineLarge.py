import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,segmentation
from organelle_measure.tools import batch_apply

def postprocess_peroxisome(path_in,path_ref,path_out):
    with h5py.File(str(path_in),'r') as f_in:
        img_in = f_in["exported_data"][:]
    img_in = np.argmax(img_in,axis=0)
    img_in = (img_in==1)

    with h5py.File(str(path_ref),'r') as f:
        img_ref = f['data'][0]
    
    # idx_max = feature.peak_local_max(img_ref,min_distance=1)
    # img_max = np.zeros_like(img_ref,dtype=bool)
    # img_max[tuple(idx_max.T)] = True

    img_out = segmentation.watershed(-img_ref,mask=img_in)
    io.imsave(
        str(path_out),
        util.img_as_uint(img_out)
    )
    return None

folder_i = "./images/preprocessed/leucine-large-blue-gaussian"
folder_o = "./images/labelled/EYrainbow_leucine_large"

list_i   = []
list_ref = []
list_o   = []
for path_in in Path(folder_i).glob("probability*.h5"):
    path_ref = Path(folder_i)/f"{path_in.stem.partition('_')[2]}.hdf5"
    path_out = Path(folder_o)/f"label-peroxisome_{path_in.stem.partition('probability_spectral-blue_')[2]}.tiff"

    list_i.append(path_in)
    list_ref.append(path_ref)
    list_o.append(path_out)
args = pd.DataFrame({
    "path_in":  list_i,
    "path_ref": list_ref,
    "path_out": list_o
})

batch_apply(postprocess_peroxisome,args)