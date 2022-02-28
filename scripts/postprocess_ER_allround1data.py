import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util
from organelle_measure.tools import skeletonize_zbyz,batch_apply

def postprocess_ER(path_in,path_out):
    img_in = io.imread(str(path_in))
    img_in = (img_in>1)
    img_ske = skeletonize_zbyz(img_in)
    io.imsave(
        str(path_out),
        util.img_as_ubyte(img_ske)
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
folder_i = "./data/preprocessed/"
fodler_o = "./data/labelled/"

list_i = []
list_o = []
for folder in folders:
    for path_binary in (Path(folder_i)/folder).glob("binary-ER*.tiff"):
        path_output = (Path(fodler_o)/folder)/f"label-{path_binary.name.partition('-')[2]}"
        list_i.append(path_binary)
        list_o.append(path_output)
args = pd.DataFrame({
    "path_in":  list_i,
    "path_out": list_o
})

batch_apply(postprocess_ER,args)
