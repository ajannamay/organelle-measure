# This script aims at finding the error that starts from the measured value
# In previous scripts, the mean of the error samples are different from the 
# meansured value, bad-looking.
# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,morphology
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
path = "/images/preprocessed/EYrainbow_glucose_largerBF/probability_mitochondria_EYrainbow_glu-100_field-3.h5"

with h5py.File(str(path)) as h5:
	image = h5["exported_data"][1]

# %%
image_int = (image*100).astype(int)
io.imsave(
	f"images/{Path(path).stem}.tif",
	util.img_as_uint(image_int)
)

# %%
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
	to_compare = 1 - prob[edge]
	randoms  = np.random.random(to_compare.shape)
	compared = (to_compare > randoms)
	mask[edge] = compared
	return None

# %%
mask_selected = (image>0.5)
N_sample = 10000

mask_up = np.copy(mask_selected)
sizes_up = np.empty(N_sample)
for i in range(N_sample):
	upsample(mask_up,image)
	sizes_up[i] = np.count_nonzero(mask_up)
	continue

mask_dw = np.copy(mask_selected)
sizes_dw = np.empty(N_sample)
for i in range(N_sample):
	continue
	


# %%
