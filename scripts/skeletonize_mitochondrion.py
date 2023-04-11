import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,morphology,measure

path_mito = "images/labelled/EYrainbow_glucose_largerBF/label-mitochondria_EYrainbow_glu-100_field-4.tiff"

img_mito = io.imread(path_mito)
skeleton3d = morphology.skeletonize((img_mito>0),method="lee")
skeleton2d = np.zeros_like(img_mito)
for z in range(skeleton2d.shape[0]):
    skeleton2d[z] = morphology.skeletonize((img_mito[z]>0),method='lee')

label_skeleton3d = measure.label(skeleton3d)
label_skeleton2d = measure.label(skeleton2d)

io.imsave("data/mito_skeleton/label_skeleton2d.tif",(label_skeleton3d))
io.imsave("data/mito_skeleton/label_skeleton3d.tif",(label_skeleton2d))

# then use scripts/measure_organelle.py:measure1organelle() 

df_original   = pd.read_csv("data/mito_skeleton/mitochondria_EYrainbow_glu-100_field-4.csv")
df_skelet2d = pd.read_csv("data/mito_skeleton/skeleton2d.csv")
df_skelet3d = pd.read_csv("data/mito_skeleton/skeleton3d.csv")

cell_original = df_original.groupby(["experiment","condition","hour","field","organelle","idx-cell"]).sum()
cell_skelet2d = df_skelet2d.groupby(["experiment","condition","hour","field","organelle","idx-cell"]).sum()
cell_skelet3d = df_skelet3d.groupby(["experiment","condition","hour","field","organelle","idx-cell"]).sum()

plt.scatter(cell_original["volume-pixel"],cell_skelet2d["volume-pixel"])
plt.scatter(cell_original["volume-pixel"],cell_skelet3d["volume-pixel"])
