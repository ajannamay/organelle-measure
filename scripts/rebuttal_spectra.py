# %%
import numpy as np
import pandas as pd
from pathlib import Path
from nd2reader import ND2Reader
from skimage import io,util
from organelle_measure.tools import read_spectral_img

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
def get_spectra(label1,label2,spectra1,spectra2):
	"""
	1 labels YFP
	2 labels Red channels
	"""
	spectra1_label1 = spectra1[label1] + 0.01    # avoid 0 for alignment & normalization
	spectra2_label1 = spectra2[label1] + 0.01    # avoid 0 for alignment & normalization
	spectra1_label2 = spectra1[label2] + 0.01    # avoid 0 for alignment & normalization
	spectra2_label2 = spectra2[label2] + 0.01    # avoid 0 for alignment & normalization

	aligned1_label1 = spectra1_label1 # / spectra1_label1[:,-1][:,np.newaxis]
	aligned2_label1 = spectra2_label1 # / spectra2_label1[:, 0][:,np.newaxis]
	aligned1_label2 = spectra1_label2 # / spectra1_label2[:,-1][:,np.newaxis]
	aligned2_label2 = spectra2_label2 # / spectra2_label2[:, 0][:,np.newaxis]

	# it seems the alignment should not be done, but why?

	spectra_label1 = np.hstack((aligned1_label1,aligned2_label1))
	spectra_label2 = np.hstack((aligned1_label2,aligned2_label2))

	max1 = np.max(spectra_label1,axis=1)[:,np.newaxis]
	max2 = np.max(spectra_label2,axis=1)[:,np.newaxis]

	normalized1 = spectra_label1/max1
	normalized2 = spectra_label2/max2

	averaged1 = np.mean(normalized1,axis=0)
	averaged1 = averaged1/np.max(averaged1)

	averaged2 = np.mean(normalized2,axis=0)
	averaged2 = averaged2/np.max(averaged2)

	return averaged1,averaged2


# %% PLOTS
fig = go.Figure()
for fov in range(1,7): # IS THIS HARD CODE NUMBER CORRECT?
	path_prob1 = Path(folder_int)/f"Probabilities_EY2796_4color_FOV{fov}_WT_yfp_unmix.tiff"
	path_prob2 = Path(folder_int)/f"Probabilities_EY2796_4color_FOV{fov}_WT_Red_unmix.tiff"
	path_spectra1 = Path(folder_raw)/f"EY2796_4color_FOV{fov}_WT_yfp.nd2"
	path_spectra2 = Path(folder_raw)/f"EY2796_4color_FOV{fov}_WT_Red.nd2"

	img_prob1 = io.imread(str(path_prob1))
	img_prob2 = io.imread(str(path_prob2))
	img_label1 = (img_prob1 > 0.5)
	img_label2 = (img_prob2 > 0.5)

	img_spectra1 = read_spectral_img(str(path_spectra1))
	img_spectra2 = read_spectral_img(str(path_spectra2))

	normalized1,normalized2 = get_spectra(img_label1,img_label2,img_spectra1,img_spectra2)
	fig.add_trace(
		go.Scatter(
			x=wl_yfp,
			y=normalized1[:10],
			name=f"FOV-{fov}-LD-yellow", mode="lines+markers",
			line = dict(dash="solid",shape="spline",color='grey')
		)
	)
	fig.add_trace(
		go.Scatter(
			x=wl_red,
			y=normalized1[10:],
			name=f"FOV-{fov}-LD-red", mode="lines+markers",
			line = dict(dash="solid",shape="spline",color='grey')
		)
	)
	
	fig.add_trace(
		go.Scatter(
			x=wl_yfp,
			y=normalized2[:10],
			name=f"FOV-{fov}-mitochondrion-yellow", mode="lines+markers",
			line = dict(dash="dash",shape="spline",color='grey')
		)
	)
	fig.add_trace(
		go.Scatter(
			x=wl_red,
			y=normalized2[10:],
			name=f"FOV-{fov}-mitochondrion-red", mode="lines+markers",
			line = dict(dash="dash",shape="spline",color='grey')
		)
	)
img_benchmark1 = np.logical_and((img_prob1>0.9),(img_prob2<0.1))
img_benchmark2 = np.logical_and((img_prob2>0.9),(img_prob1<0.1))
benchmark1,benchmark2 = get_spectra(img_benchmark1,img_benchmark2,img_spectra1,img_spectra2)
fig.add_trace(
	go.Scatter(
		x = wl_yfp,
		y = benchmark1[:10],
		name="lipid droplet", mode="lines+markers",
		line=dict(dash="solid",shape="spline",color="blue")
	)
)
fig.add_trace(
	go.Scatter(
		x = wl_red,
		y = benchmark1[10:],
		name="lipid droplet", mode="lines+markers",
		line=dict(dash="solid",shape="spline",color="blue")
	)
)

fig.add_trace(
	go.Scatter(
		x = wl_yfp,
		y = benchmark2[:10],
		name="mitochondrion", mode="lines+markers",
		line=dict(dash="solid",shape="spline",color="red")
	)
)
fig.add_trace(
	go.Scatter(
		x = wl_red,
		y = benchmark2[10:],
		name="mitochondrion", mode="lines+markers",
		line=dict(dash="solid",shape="spline",color="red")
	)
)
fig.update_layout(template="simple_white")
fig.write_html(f"data/rebuttal_spectra/from_Kiandokht_strain.html")

# %% END BIOFORMATS, USE ONCE, OTHERWISE NEED TO RESTART KERNEL
javabridge.kill_vm()

# %%
