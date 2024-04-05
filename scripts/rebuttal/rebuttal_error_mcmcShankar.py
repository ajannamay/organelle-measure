# This script aims at finding the error that starts from the measured value
# In previous scripts, the mean of the error samples are different from the 
# meansured value, bad-looking.
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,morphology,measure
import h5py

organelles = [
    "peroxisome",
    "vacuole",
    "ER",
    "golgi",
    "mitochondria",
    "LD"

] 


# %%
# both functions do not have return values, i.e.:
# their executions are not expected to be assigned to some variables
# instead, the first input will be altered by the functions.
def upsample(mask,prob):
    dilated = morphology.binary_dilation(mask)
    edge = np.logical_xor(mask,dilated)
    to_compare = prob[edge]
    randoms  = np.random.random(to_compare.shape)
    compared = (to_compare > randoms)
    mask[edge] = compared
    return None

def downsample(mask,prob):
    eroded = morphology.binary_erosion(mask)
    edge = np.logical_xor(mask,eroded)
    to_compare = prob[edge] # not (1 - prob[edge]), because last line means a flip
    randoms  = np.random.random(to_compare.shape)
    compared = (to_compare > randoms)
    mask[edge] = compared
    return None

# %%
N_sample = 20000
stem = "EYrainbow_glu-100_field-3"

path_cell = f"images/cell/EYrainbow_glucose_largerBF/binCell_{stem}.tif"
img_cell = io.imread(str(path_cell))

dfs = []
for organelle in organelles:
    print(f"Start processing: {organelle}")
    path_organelle = f"images/preprocessed/EYrainbow_glucose_largerBF/probability_{organelle}_{stem}.h5"
    with h5py.File(str(path_organelle)) as h5:
        img_orga = h5["exported_data"][1]

    index = []
    trues = []
    means = []
    stdvs = []
    for cell in measure.regionprops(img_cell):
        min_row, min_col, max_row, max_col = cell.bbox
        img_cell_crop = cell.image
        
        img_orga_crop = img_orga[:,min_row:max_row,min_col:max_col]
        for z in range(img_orga_crop.shape[0]):
            img_orga_crop[z] = img_orga_crop[z] * img_cell_crop

        mask_selected = (img_orga_crop > 0.5)
        mask_dynamic  = np.copy(mask_selected)
        sizes = np.empty(N_sample)
        sizes[0] = np.count_nonzero(mask_selected)
        for i in range(N_sample-1):
            seed = np.random.random()
            if seed < 0.5:
                downsample(mask_dynamic,img_orga_crop)
            else:
                upsample(  mask_dynamic,img_orga_crop)
            sizes[i+1] = np.count_nonzero(mask_dynamic)
    
        index.append(cell.label)
        trues.append(sizes[0])
        means.append(sizes.mean())
        stdvs.append(sizes.std())

        print(f"... simulated cell #{cell.label}")
    dfs.append(pd.DataFrame({
        "organelle": organelle,
        "index"    : index,
        "segmented": trues,
        "average"  : means,
        "standard_deviation": stdvs
    }))
    print(f"Finished: {organelle}")

    
df = pd.concat(dfs,ignore_index=True)
df.to_csv("plots/rebuttal_error/mcmcShankar.csv",index=False)

        

# %%
