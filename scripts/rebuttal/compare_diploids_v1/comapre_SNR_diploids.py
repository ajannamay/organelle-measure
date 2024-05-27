import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,util
from nd2reader import ND2Reader

organelles = [
	"peroxisome",
	# "vacuole",
	"ER",
	"golgi",
	"mitochondria",
	"LD"
]
meta = {
	"peroxisome": {
		"prefix": "unmixed-blue",
		"c"     : 0
	},
	"vacuole": {
		"prefix": "unmixed-blue",
		"c"     : 1
	},
	"ER": {
		"prefix": "spectral-green"
	},
	"golgi": {
		"prefix": "spectral-yellow",
		"c": -1
	},
	"mitochondria": {
		"prefix": "unmixed-red",
		"c"     : 0
	},
	"LD": {
		"prefix": "unmixed-red",
		"c"     : 0
	},
}


# # Plot mean signal and SNR
signal_noise = {o:{} for o in organelles}

def average_intensities(img_intensity,img_probability):
	img_mask = (img_probability>0.5)
	img_edge = np.logical_and(img_probability>0.495, img_probability<0.505)
	mean_signal = np.mean(img_intensity[img_mask])
	mean_backgd = np.mean(img_intensity[np.logical_not(img_mask)])
	mean_edge   = np.mean(img_intensity[img_edge])
	return mean_signal,mean_backgd,mean_edge

# ## 6-color original experimental data
for organelle in organelles:
	for path_cell in Path("images/cell/EYrainbow_glucose_largerBF").glob("*glu-100*.tif"):
		stem = path_cell.stem.partition("_")[2]
		path_mask = Path("images/preprocessed/EYrainbow_glucose_largerBF")/f"probability_{organelle}_{stem}.h5"
		path_raw  = Path("images/raw/EYrainbow_glucose_largerBF")/f"{meta[organelle]['prefix']}_{stem}.nd2"
		# print(organelle,path_mask.exists(),path_mask.name,path_raw.exists(),path_raw.name)
		img_raw = 0
		with ND2Reader(str(path_raw)) as nd2:
			# print(organelle,stem,nd2.sizes)
			if "c" in meta[organelle].keys():
				if meta[organelle]["c"] == -1:
					# print(organelle)
					nd2.iter_axes = 't'
					nd2.bundle_axes = "czyx"
					img_raw = nd2[0]
					img_raw = np.sum(img_raw,axis=0)
				else:
					nd2.iter_axes = 'c'
					nd2.bundle_axes = "zyx"
					img_raw = nd2[meta[organelle]['c']]
			else:
				nd2.iter_axes = 't'
				nd2.bundle_axes = "zyx" 
				img_raw = nd2[0]
		with h5py.File(str(path_mask),'r') as h5:
			img_prob = h5["exported_data"][1]
		# print(organelle,stem,img_raw.shape,img_mask.shape)
		mean_signal,mean_backgd,mean_edge = average_intensities(img_raw, img_prob)
		signal_noise[organelle]["6color-3hour"] = {
													"signal": mean_signal,
													"noise" : mean_backgd,
													"edge"  : mean_edge
												}

# ## 6-color 8-hour data
for organelle in organelles:
	for path_cell in Path("images/cell/paperRebuttal").glob("*glu-100*.tif"):
		stem = path_cell.stem.partition("_")[2]
		path_mask = Path("images/preprocessed/paperRebuttal")/f"probability_{organelle}_{stem}.h5" # why is this not tiff?
		path_raw  = Path("images/raw/paperRebuttal")/f"{meta[organelle]['prefix']}_{stem}.nd2"
		# print(organelle,path_mask.exists(),path_mask.name,path_raw.exists(),path_raw.name)
		img_raw = 0
		with ND2Reader(str(path_raw)) as nd2:
			# print(organelle,stem,nd2.sizes)
			if "c" in meta[organelle].keys():
				if meta[organelle]["c"] == -1:
					# print(organelle)
					nd2.iter_axes = 't'
					nd2.bundle_axes = "zyx"
					img_raw = nd2[0]
				else:
					nd2.iter_axes = 'c'
					nd2.bundle_axes = "zyx"
					img_raw = nd2[meta[organelle]['c']]
			else:
				nd2.iter_axes = 't'
				nd2.bundle_axes = "zyx" 
				img_raw = nd2[0]
		with h5py.File(str(path_mask),'r') as h5:
			img_mask = h5["exported_data"][1]
		# print(organelle,stem,img_raw.shape,img_mask.shape)
		img_mask = (img_mask>0.5)
		mean_signal = np.mean(img_raw[img_mask])
		mean_backgd = np.mean(img_raw[np.logical_not(img_mask)])
		signal_noise[organelle]["6color-8hour"] = {"signal":mean_signal,"noise":mean_backgd}

# ## 1-color data
raw1color = {
	"peroxisome": {
		"raw": "px_unmixed-blue_diploid1color_field-1.nd2",
		"c"  : 0
	},
	# "vacuole": {},
	"ER": {
		"raw": "er_spectra-green_diploid1color_field-1.nd2"
	},
	"golgi": {
		"raw": "gl_spectra-yellow_diploid1color_field-1.nd2"
	},
	"mitochondria": {
		"raw": "mt_spectral-red_diploid1color_field-1.nd2",
		"c"  : 0
	},
	"LD": {
		"raw": "ld_unmixed-red_diploid1color_field-1.nd2",
		"c"  : 1
	}
}
for organelle in organelles:
	path_mask = Path("images/preprocessed/2024-02-16_rebuttal1color")/f"SameParamAs6_preprocessed_{organelle}_diploids1color.tiff"
	path_raw  = Path("images/raw/2024-02-16_rebuttal1color")/raw1color[organelle]["raw"]
	
	if "c" in raw1color[organelle].keys():
		continue

# # Plot 