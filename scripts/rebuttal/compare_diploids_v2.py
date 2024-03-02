import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,segmentation,measure,filters
from nd2reader import ND2Reader
from organelle_measure.yeaz import yeaz_preprocesses,yeaz_label
from organelle_measure.tools import skeletonize_zbyz,neighbor_mean,find_complete_rings


# Segment cells, only need to do 6-color cells
for path in Path("images/raw/2024-02-16_rebuttal1color").glob("all_camera*.nd2"):
	with ND2Reader(str(path)) as nd2:
		nd2.bundle_axes = 'yx'
		nd2.iter_axes = 't'
		img = nd2[0]
	for prep in yeaz_preprocesses:
		img = prep(img)
	segmented = yeaz_label(img, min_dist=5)
	segmented = segmentation.clear_border(segmented)
	properties = measure.regionprops(segmented)
	for prop in properties:
		if prop.area < 50:
			segmented[segmented==prop.label] = 0
	segmented = measure.label(segmented)
	output = np.zeros((512,512),dtype=int)
	shape0,shape1 = segmented.shape
	output[:shape0,:shape1] = segmented

	io.imsave(
		f"images/cell/2024-02-16_rebuttal1color/{path.stem}.tif",
		util.img_as_uint(output)
	)


# Preprocess, only 6-color images, 1-color already processed 

# peroxisome
path = Path(f"images/raw/2024-02-16_rebuttal1color/all_unmixed-blue_diploid6color_field-1.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "zyx"
	nd2.iter_axes   = "c"
	image = nd2[0]
gauss = filters.gaussian(image,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/2024-02-16_rebuttal1color/preprocessed_peroxisome_diploids6color.tif",
	util.img_as_uint(gauss)
)

# vacuole
path = Path(f"images/raw/2024-02-16_rebuttal1color/all_unmixed-blue_diploid6color_field-1.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "zyx"
	nd2.iter_axes   = "c"
	image = nd2[1]
gauss = filters.gaussian(image,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/2024-02-16_rebuttal1color/preprocessed_vacuole_diploids6color.tif",
	util.img_as_uint(gauss)
)

# ER
path = Path(f"images/raw/2024-02-16_rebuttal1color/all_spectra-green_diploid6color_field-1.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = 'zyx'
	nd2.iter_axes = "t"
	image = nd2[0]
gauss = filters.gaussian(image,sigma=0.3,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/2024-02-16_rebuttal1color/preprocessed_ER_diploids6color.tif",
	util.img_as_uint(gauss)
)

# golgi
path = Path(f"images/raw/2024-02-16_rebuttal1color/all_spectra-yellow_diploid6color_field-1.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "zyx"
	nd2.iter_axes = 't'
	image = nd2[0]
gauss = filters.gaussian(image,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/2024-02-16_rebuttal1color/preprocessed_golgi_diploids6color.tif",
	util.img_as_uint(gauss)
)

# mitochondria
path_cell = Path(f"images/cell/2024-02-16_rebuttal1color/all_camera-after_diploid6color_field-1.tif")
cell = io.imread(str(path_cell))
path = Path(f"images/raw/2024-02-16_rebuttal1color/all_unmixed-red_diploid6color_field-1.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "zyx"
	nd2.iter_axes = 'c'
	image = nd2[0]
bkgd = neighbor_mean(image,cell)
clear = image - bkgd
clear[clear<0] = 0
gauss = filters.gaussian(clear,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/2024-02-16_rebuttal1color/preprocessed_mitochondria_diploids6color.tif",
	util.img_as_uint(gauss)
)

# LD
path_cell = Path(f"images/cell/2024-02-16_rebuttal1color/all_camera-after_diploid6color_field-1.tif")
cell = io.imread(str(path_cell))
path = Path(f"images/raw/2024-02-16_rebuttal1color/all_unmixed-red_diploid6color_field-1.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "zyx"
	nd2.iter_axes = 'c'
	image = nd2[1]
bkgd = neighbor_mean(image,cell)
clear = image - bkgd
clear[clear<0] = 0
gauss = filters.gaussian(clear,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/2024-02-16_rebuttal1color/preprocessed_LD_diploids6color.tif",
	util.img_as_uint(gauss)
)


# Postprocess, after Ilastik

# peroxisome, golgi, lipid droplet (globular organelles)

