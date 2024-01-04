# %%
import numpy as np
import pandas as pd
from pathlib import Path
from nd2reader import ND2Reader
from skimage import io,util

import plotly.express as px
import plotly.graph_objects as go


# We're trying to prove the red channels separate 
# mitochondria and lipid droplets correctly
# by using a strain that has a different color for LDs(?)
folder_raw = "images/raw/Kiandohkt4colorWT"
folder_int = "images/preprocessed/rebuttal_spectra"

# wavelengths of different acquisitions
wl_yfp = [511,517,523,529,535,541,547,553,559,565]
wl_red = [568,574,580,586,592,598,604,610,616,622,628,634,640,646,652]

# %% 
# ND2READER DOES NOT WORK FOR IMAGES >7 CHANNELS
for path in Path(folder_raw).glob("*unmix.nd2"):
	with ND2Reader(str(path)) as file_img:
		file_img.bundle_axes = "zyx"
		file_img.iter_axes = 't'
		img = file_img[0]
	io.imsave(
		str(Path(folder_int)/f"{path.stem}.tif"),
		util.img_as_uint(img.astype(int))
	)
# Then the images were segmented with `ilastik`


# %% START BIOFORMATS, USE ONCE
import javabridge
import bioformats

javabridge.start_vm(class_path=bioformats.JARS)

# %%
def open_spectral_img(path):
	with ND2Reader(str(path)) as nd2_img:
		size_img = nd2_img.sizes

	sample_img  = bioformats.load_image(
					str(path), 
					c=None, z=0, t=0, series=None, index=None,
					rescale=False, wants_max_intensity=False, 
					channel_names=None
	              )
	array_img = np.empty((size_img['z'],*sample_img.shape))
	for z in range(size_img['z']):
		array_img[z] = bioformats.load_image(
							str(path), 
							c=None, z=z, t=0, series=None, index=None,
							rescale=False, wants_max_intensity=False, 
							channel_names=None
	            	   )
	return array_img

path_img = Path(folder_raw)/"EY2796_4color_FOV1_WT_bfp.nd2"
test = open_spectral_img(path_img)

# %%
# 1 labels YFP
# 2 labels Red channels

# Extract the spectra, for each pixel in an FOV
path_label1 = ""
path_label2 = ""
path_spectra1 = ""
path_spectra2 = ""

img_label1 = io.imread(str(path_label1))
img_label2 = io.imread(str(path_label2))
img_label1 = (img_label1 > 0.5)
img_label2 = (img_label2 > 0.5)

img_spectra1 = open_spectral_img(str(path_spectra1))
img_spectra2 = open_spectral_img(str(path_spectra2))

spectra1_label1 = img_spectra1[img_label1]
spectra2_label1 = img_spectra2[img_label1]
spectra1_label2 = img_spectra1[img_label2]
spectra2_label2 = img_spectra2[img_label2]

spectra_label1 = np.hstack((spectra1_label1,spectra2_label1))
spectra_label2 = np.hstack((spectra1_label2,spectra2_label2))

# TODO: Normalize and Average
normalized1 = []
normalized2 = []

# TODO: Get the benchmark from non-overlaping spectra
benchmark1 = []
benchmark2 = []

# %% PLOTS
fig = go.Figure()
for fov in range(1,5): # IS THIS HARD CODE NUMBER CORRECT?
	normalized1 = np.array()
	normalized2 = np.array()
	fig.add_trace(
		go.Scatter(
			x=wl_yfp+wl_red,
			y=normalized1,
			name=f"FOV-{fov}", mode="lines+markers",
			line = dict(dash="solid",shape="spline",color='grey')
		)
	) 
	fig.add_trace(
		go.Scatter(
			x=wl_yfp+wl_red,
			y=normalized2,
			name=f"FOV-{fov}", mode="lines+markers",
			line = dict(dash="dash",shape="spline",color='grey')
		)
	) 
fig.add_trace(
	go.Scatter(
		x = wl_yfp + wl_red,
		y = benchmark1,
		name="lipid droplet", mode="lines+markers",
		line=dict(dash="solid",shape="spine",color="blue")
	)
)
fig.add_trace(
	go.Scatter(
		x = wl_yfp + wl_red,
		y = benchmark2,
		name="mitochondrion", mode="lines+markers",
		line=dict(dash="solid",shape="spine",color="red")
	)
)
fig.update_layout(template="simple_white")
fig.write_html(f"data/rebuttal_spectra/from_Kiandokht_strain.html")

# %% END BIOFORMATS, USE ONCE, OTHERWISE NEED TO RESTART KERNEL
javabridge.kill_vm()

# %%
