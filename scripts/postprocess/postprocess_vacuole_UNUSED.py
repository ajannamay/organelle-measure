import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,morphology,measure
from organelle_measure.tools import skeletonize_zbyz,watershed_zbyz,find_complete_rings,better_vacuole_img,batch_apply

def postprocess_vacuole(path_in,path_cell,path_out):
    img_orga = io.imread(str(path_in))
    img_orga = (img_orga>1)

    # img_cell = io.imread(str(path_cell))

    img_skeleton  = skeletonize_zbyz(img_orga)

    img_core = find_complete_rings(img_skeleton)
    # img_watershed = watershed_zbyz(img_skeleton)

    # img_vacuole   = better_vacuole_img(img_core,img_watershed)

    # img_vacuole = np.zeros_like(img_watershed,dtype=int)
    # for z in range(img_watershed.shape[0]):
    #     slice_watershed = img_watershed[z]
    #     candidates = np.unique(slice_watershed[img_cell>0])
    #     for color in candidates:
    #         if len(np.unique(img_cell[slice_watershed==color]))==1:
    #             img_vacuole[z,slice_watershed==color] = color


    # img_vacuole   = better_vacuole_img(img_core,img_watershed)
    io.imsave(
        str(path_out),
        util.img_as_uint(img_core) 
    )
    return None

folders = [
    "EYrainbow_glucose",
    "EYrainbow_glucose_largerBF",
    "EYrainbow_rapamycin_1stTry",
    "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st",
    "EYrainbowWhi5Up_betaEstrodiol",
    # "EYrainbow_leucine_large",
    "EYrainbow_leucine"
]
folder_i = "./data/preprocessed/"
folder_c = "./data/cell/"
folder_o = "./test/vacuole2/"

print("Scanning folders...")
for folder in folders:
    if (newfolder:=Path(folder_o)/folder).exists():
        print(f"Folder `{newfolder.name}` exists. Skip...")
    else:
        newfolder.mkdir()
        print(f"Folder `{newfolder.name}` created. Next...")

list_i = []
list_c = []
list_o = []
for folder in folders:
    count = 0
    for path_cell in (Path(folder_c)/folder).glob(f"*.tif"):
        path_binary = (Path(folder_i)/folder)/f"*binary-vacuole_{path_cell.stem.partition('_')[2]}.tiff"
        path_output = (Path(folder_o)/folder)/f"core-{path_binary.name.partition('-')[2]}"
        list_i.append(path_binary)
        list_c.append(path_cell)
        list_o.append(path_output)
        count += 1
        if count>4:
            break
args = pd.DataFrame({
    "path_in":   list_i,
    "path_cell": list_c,
    "path_out":  list_o
})

batch_apply(postprocess_vacuole,args)
