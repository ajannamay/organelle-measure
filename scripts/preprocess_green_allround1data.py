import numpy as np
import pandas as pd
from pathlib import Path
from skimage import util,io,filters
from organelle_measure.tools import open_organelles,neighbor_mean,batch_apply
from organelle_measure.vars_allround1data import list_folders

def preprocess_green(path_in,path_out,organelle):
    img_raw   = open_organelles[organelle](str(path_in))
    img_gaussian = filters.gaussian(img_raw,sigma=0.3,preserve_range=True).astype(int)
    io.imsave(str(path_out),util.img_as_uint(img_gaussian))
    return None

list_in   = []
list_out  = []
list_orga = []
for folder in ["EYrainbow_leucine_large"]:
    for path_in in (Path("./data/raw")/folder).glob("spectral-green*.nd2"):
        path_ER = Path("./data/preprocessed")/folder/f'ER_{path_in.stem.partition("_")[2]}.tif'
        list_in.append(path_in)
        list_out.append(path_ER)
        list_orga.append("ER")

args = pd.DataFrame({
    "path_in": list_in,
    "path_out": list_out,
    "organelle": list_orga
})

batch_apply(preprocess_green,args)