# This script aims at finding the error that starts from the measured value
# In previous scripts, the mean of the error samples are different from the 
# meansured value, bad-looking.
# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util
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
path = r"D:\Documents\GitHub\ShixingWang\organelle-measure\images\preprocessed\EYrainbow_glucose_largerBF\probability_mitochondria_EYrainbow_glu-100_field-3.h5"

with h5py.File(str(path)) as h5:
	image = h5["exported_data"][1]

# %%
image_int = (image*100).astype(int)
io.imsave(
	f"images/{Path(path).stem}.tif",
	util.img_as_uint(image_int)
)

# %%
mask_selected = (image>0.5)

N_sample = 10000
sizes = np.empty(N_sample)
for i in range(N_sample):
	


# %%
