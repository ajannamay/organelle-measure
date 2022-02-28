import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,segmentation
from organelle_measure.tools import batch_apply

def postprocess_globular(path_in,path_ref,path_out):
    img_in = io.imread(str(path_in))
    img_in = (img_in>1)
    img_ref = io.imread(str(path_ref))
    
    # idx_max = feature.peak_local_max(img_ref,min_distance=1)
    # img_max = np.zeros_like(img_ref,dtype=bool)
    # img_max[tuple(idx_max.T)] = True

    img_out = segmentation.watershed(-img_ref,mask=img_in)
    io.imsave(
        str(path_out),
        util.img_as_uint(img_out)
    )
    return None

folders = [
    "EYrainbow_glucose",
    "EYrainbow_glucose_largerBF",
    "EYrainbow_rapamycin_1stTry",
    "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st",
    "EYrainbowWhi5Up_betaEstrodiol",
    "EYrainbow_leucine_large",
    "EYrainbow_leucine"
]
organelles = ["peroxisome","LD","golgi"]
folder_i = "./data/preprocessed/"
fodler_o = "./data/labelled/"


list_i   = []
list_ref = []
list_o   = []
for folder in folders:
    for organelle in organelles:
        for path_binary in (Path(folder_i)/folder).glob(f"binary-{organelle}*.tiff"):
            path_output = (Path(fodler_o)/folder)/f"label-{path_binary.name.partition('-')[2]}"
            path_ref = (Path(folder_i)/folder)/f"{path_binary.stem.partition('-')[2]}.tif"
            list_i.append(path_binary)
            list_ref.append(path_ref)
            list_o.append(path_output)
args = pd.DataFrame({
    "path_in":  list_i,
    "path_ref": list_ref,
    "path_out": list_o
})

batch_apply(postprocess_globular,args)