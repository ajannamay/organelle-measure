import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,segmentation,measure
from organelle_measure.tools import skeletonize_zbyz

FOLDER1 = "2024-02-16_rebuttal1color"
FOLDER6 = "EYrainbow_glucose_largerBF"
FOLDER  = "2024-02-21_rebuttal1n6color"

for folder in [
	Path(f"images/labelled/{FOLDER}/ilastik-1_color-6"),
	Path(f"images/labelled/{FOLDER}/ilastik-6_color-1"),
	Path(f"data/results/{FOLDER}/ilastik-1_color-6"),
	Path(f"data/results/{FOLDER}/ilastik-6_color-1")
]:
	if not folder.exists():
		Path.mkdir(folder)

# (no need to do) Segment Cells 
# (no need to do) Preprocess

# after Ilastik, Postprocess
# peroxisome, golgi, lipid droplet
for path in [
	Path(f"images/preprocessed/{FOLDER1}/SixColorProbability_preprocessed_peroxisome_diploids1color.tiff"),
	Path(f"images/preprocessed/{FOLDER1}/SixColorProbability_preprocessed_golgi_diploids1color.tiff"),
	Path(f"images/preprocessed/{FOLDER1}/SixColorProbability_preprocessed_LD_diploids1color.tiff")
]:
	img = io.imread(str(path))
	img = (img>0.5)
	path_ref = path.parent / f"{path.stem.partition('_')[2]}.tif"
	ref = io.imread(str(path_ref))
	output = segmentation.watershed(-ref,mask=img)
	io.imsave(
		f"images/labelled/{FOLDER}/ilastik-6_color-1/{path_ref.name.partition('preprocessed_')[2]}",
		util.img_as_uint(output)
	)

for organelle in ["peroxisome","golgi","LD"]:
	for path in Path(f"images/preprocessed/{FOLDER6}/").glob(f"SingleColorProbability_{organelle}*.tiff"):
		img = io.imread(str(path))
		img = (img>0.5)
		path_ref = path.parent / f"{path.stem.partition('_')[2]}.tif"
		ref = io.imread(str(path_ref))
		output = segmentation.watershed(-ref,mask=img)
		io.imsave(
			f"images/labelled/{FOLDER}/ilastik-1_color-6/{path_ref.name}",
			util.img_as_uint(output)
		)

# mitochondria
path = Path(f"images/preprocessed/{FOLDER1}/SixColorProbability_preprocessed_mitochondria_diploids1color.tiff")
img = io.imread(str(path))
img = (img>0.5)
output = measure.label(img)
io.imsave(
	f"images/labelled/{FOLDER}/ilastik-6_color-1/mitochondria_diploids1color.tif",
	util.img_as_uint(output)
)

for path in Path(f"images/preprocessed/{FOLDER6}").glob("SingleColorProbability_mitochondria*.tiff"):
	img = io.imread(str(path))
	img = (img>0.5)
	output = measure.label(img)
	io.imsave(
		f"images/labelled/{FOLDER}/ilastik-1_color-6/{path.stem.partition('_')[2]}.tif",
		util.img_as_uint(output)
	)

# ER
path = Path(f"images/preprocessed/{FOLDER1}/SixColorProbability_preprocessed_ER_diploids1color.tiff")
img = io.imread(str(path))
img = (img>0.5)
ske = skeletonize_zbyz(img)
io.imsave(
	f"images/labelled/{FOLDER}/ilastik-6_color-1/ER_diploids1color.tif",
	util.img_as_uint(ske)
)

for path in Path(f"images/preprocessed/{FOLDER6}").glob("SingleColorProbability_ER*.tiff"):
	img = io.imread(str(path))
	img = (img>0.5)
	ske = skeletonize_zbyz(img)
	io.imsave(
		f"images/labelled/{FOLDER}/ilastik-1_color-6/{path.stem.partition('_')[2]}.tif",
		util.img_as_uint(ske)
	)


# # vacuole
# path = Path(f"images/preprocessed/{FOLDER}/Probabilities_preprocessed_vacuole_diploids1color.tiff")
# img = io.imread(str(path))
# img = (img>0.5)
# path_cell = Path(f"images/cell/{FOLDER}/camera-BF-after_EY2795triColor-EY2796WT_check-2.tif")
# cell = io.imread(path_cell)
# ske = skeletonize_zbyz(img)
# core = find_complete_rings(ske)
# output = np.zeros_like(core,dtype=int)
# for z in range(output.shape[0]):
# 	sample = core[z]
# 	candidates = np.unique(sample[cell>0])
# 	for color in candidates:
# 		if len(np.unique(cell[sample==color]))==1:
# 			output[z,sample==color] = color
# io.imsave(
# 	f"images/labelled/{FOLDER}/vacuole_diploids1color.tif",
# 	util.img_as_uint(output)
# )

# Measure
organelles = [
	"peroxisome",
	# "vacuole",
	"ER",
	"golgi",
	"mitochondria",
	"LD"
]

# single-color
imgs_cell = {
	"peroxisome"  : "px_camera-before_diploid1color_field-1.tif",
	# "vacuole"     : "vo_camera-after_diploid1color_field-1.tif",
	"ER"          : "er_camera-after_diploid1color_field-1.tif",
	"golgi"       : "gl_camera-after_diploid1color_field-1.tif",
	"mitochondria": "mt_camera-after_diploid1color_field-1.tif",
	"LD"          : "ld_camera-after_diploid1color_field-1.tif"
}
for organelle in organelles:
	path_cell      = Path(f"images/cell/{FOLDER1}/{imgs_cell[organelle]}")
	path_organelle = Path(f"images/labelled/{FOLDER}/ilastik-6_color-1/{organelle}_diploids1color.tif")
	
	img_cell      = io.imread(str(path_cell))
	img_organelle = io.imread(str(path_organelle))

	results = []
	measured = {"organelle": organelle}
	for cell in measure.regionprops(img_cell):
		measured["cell-idx"] = cell.label
		measured["cell-area"] = cell.area

		min_row, min_col, max_row, max_col = cell.bbox
		img_cell_crop = cell.image
		img_orga_crop = img_organelle[:,min_row:max_row,min_col:max_col]
		for z in range(img_orga_crop.shape[0]):
			img_orga_crop[z] = img_orga_crop[z] * img_cell_crop
		if not organelle == "vacuole":
			measured_orga = measure.regionprops_table(
				img_orga_crop,
				properties=('label','area','bbox_area','bbox')
			)
		else:
			vacuole_area = 0
			vacuole_bbox_area = 0
			bbox0,bbox1,bbox2,bbox3,bbox4,bbox5 = 0,0,0,0,0,0
			for z in range(img_orga_crop.shape[0]):
				vacuole = measure.regionprops_table(
                    img_orga_crop[z],
                    properties=('label','area','bbox_area','bbox')
                )
				if len(vacuole["area"]) == 0:
					continue
				if (maxblob:=max(vacuole["area"])) > vacuole_area:
					vacuole_area = maxblob
					idxblob = np.argmax(vacuole["area"])
					vacuole_bbox_area = vacuole["bbox_area"][idxblob]
					bbox0,bbox3 = z,z
					bbox1,bbox2,bbox4,bbox5 = [vacuole[f"bbox-{i}"][idxblob] for i in range(4)]
			if vacuole_area==0:
				continue
			measured_orga = {
				'label': [0],
                'area':  [vacuole_area],
                "bbox_area": [vacuole_bbox_area],
                "bbox-0": [bbox0],
                "bbox-1": [bbox1],
                "bbox-2": [bbox2],
                "bbox-3": [bbox3],
                "bbox-4": [bbox4],
                "bbox-5": [bbox5],
			}
		measured = measured | measured_orga
		results.append(pd.DataFrame(measured))
	result = pd.concat(results,ignore_index=True)
	result.to_csv(f"data/results/{FOLDER}/ilastik-6_color-1/{organelle}.csv",index=False)
# six-color
for path_cell in Path(f"images/cell/{FOLDER6}").glob(f"binCell_EYrainbow_glu-100_field-*.tif"):
	for organelle in organelles:
		path_organelle = Path(f"images/labelled/{FOLDER}/ilastik-1_color-6/{organelle}_{path_cell.stem.partition('_')[2]}.tif")
		
		img_cell      = io.imread(str(path_cell))
		img_organelle = io.imread(str(path_organelle))

		results = []
		measured = {"organelle": organelle}
		measured["field"] = path_organelle.stem.partition("field-")[2].partition('_')[0]
		for cell in measure.regionprops(img_cell):
			measured["cell-idx"] = cell.label
			measured["cell-area"] = cell.area

			min_row, min_col, max_row, max_col = cell.bbox
			img_cell_crop = cell.image
			img_orga_crop = img_organelle[:,min_row:max_row,min_col:max_col]
			for z in range(img_orga_crop.shape[0]):
				img_orga_crop[z] = img_orga_crop[z] * img_cell_crop
			if not organelle == "vacuole":
				measured_orga = measure.regionprops_table(
					img_orga_crop,
					properties=('label','area','bbox_area','bbox')
				)
			else:
				vacuole_area = 0
				vacuole_bbox_area = 0
				bbox0,bbox1,bbox2,bbox3,bbox4,bbox5 = 0,0,0,0,0,0
				for z in range(img_orga_crop.shape[0]):
					vacuole = measure.regionprops_table(
						img_orga_crop[z],
						properties=('label','area','bbox_area','bbox')
					)
					if len(vacuole["area"]) == 0:
						continue
					if (maxblob:=max(vacuole["area"])) > vacuole_area:
						vacuole_area = maxblob
						idxblob = np.argmax(vacuole["area"])
						vacuole_bbox_area = vacuole["bbox_area"][idxblob]
						bbox0,bbox3 = z,z
						bbox1,bbox2,bbox4,bbox5 = [vacuole[f"bbox-{i}"][idxblob] for i in range(4)]
				if vacuole_area==0:
					continue
				measured_orga = {
					'label': [0],
					'area':  [vacuole_area],
					"bbox_area": [vacuole_bbox_area],
					"bbox-0": [bbox0],
					"bbox-1": [bbox1],
					"bbox-2": [bbox2],
					"bbox-3": [bbox3],
					"bbox-4": [bbox4],
					"bbox-5": [bbox5],
				}
			measured = measured | measured_orga
			results.append(pd.DataFrame(measured))
		result = pd.concat(results,ignore_index=True)
		result.to_csv(f"data/results/{FOLDER}/ilastik-1_color-6/{path_organelle.stem}.csv",index=False)