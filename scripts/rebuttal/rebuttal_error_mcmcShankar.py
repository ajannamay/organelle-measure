# This script aims at finding the error that starts from the measured value
# In previous scripts, the mean of the error samples are different from the 
# meansured value, bad-looking.
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
path = "images/preprocessed/EYrainbow_glucose_largerBF/probability_peroxisome_EYrainbow_glu-100_field-3.h5"

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
	to_compare = prob[edge] # not (1 - prob[edge]), because last line means a flip
	randoms  = np.random.random(to_compare.shape)
	compared = (to_compare > randoms)
	mask[edge] = compared
	return None

# %%
img = image
# %%
# for z in range(image.shape[0]):
# 	img = image[z]
	mask_selected = (img>0.5)
	N_sample = 1000

	sizes = np.empty(N_sample)
	mask_dynamic = np.copy(mask_selected)
	sizes[0] = np.count_nonzero(mask_selected)
	for i in range(N_sample-1):
		seed = np.random.random()
		if seed < 0.5:
			downsample(mask_dynamic,img)
			upsample(  mask_dynamic,img)
		else:
			upsample(  mask_dynamic,img)
			downsample(mask_dynamic,img)
		sizes[i+1] = np.count_nonzero(mask_dynamic)
	plt.figure()
	plt.scatter(np.arange(len(sizes)),sizes)
	plt.plot([0,N_sample],[sizes[0],sizes[0]],'r')
	# plt.title(f"{z=}")
	plt.show()

# %%
