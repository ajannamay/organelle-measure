# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from skimage import io,util,measure,segmentation
from organelle_measure.yeaz import yeaz_label
from nd2reader import ND2Reader

# %%
folder_i = Path("images/raw/EYrainbow_glucose_largerBF")
folder_o = Path("images/cell/EYrainbow_glucose_largerBF_spectral")
folder_c = Path("images/cell/EYrainbow_glucose_largerBF")

# # %% This is not used. Do as the paper says.
# for path_i in folder_i.glob("*green*.nd2"):
# 	with ND2Reader(str(path_i)) as nd2:
# 		nd2.bundle_axes = "zyx"
# 		nd2.iter_axes   = "t"
# 		stack = nd2[0]
# 	img_i = np.max(stack,axis=0)
# 	img_i = (img_i - img_i.min())/(img_i.max() - img_i.min())
# 	img_b = yeaz_label(img_i,min_dist=5)
# 	img_b = segmentation.clear_border(img_b)
# 	properties = measure.regionprops(img_b)
# 	for prop in properties:
# 		if prop.area < 50: # hard coded threshold, bad
# 			img_b[img_b==prop.label] = 0
# 	img_b = measure.label(img_b)
	
# 	path_o = folder_o/f"binCell_{path_i.stem.partition('_')[2]}.tif"
# 	io.imsave(
# 		str(path_o),
# 		util.img_as_uint(img_b)
#   )

# %%
for path_bool in Path(folder_o).glob("ER_EYrainbow_glu-100_field-*.tif"):
	img_bool = io.imread(str(path_bool))
	img_bool = segmentation.clear_border(img_bool)
	img_int = measure.label(img_bool)
	io.imsave(
		str(folder_o/f"binCell_{path_bool.name.partition('_')[2]}"),
		util.img_as_uint(img_int)
	)
# %%
size_camera   = []
size_spectral = []
for path_camera in folder_c.glob("binCell_EYrainbow_glu-100_field-*.tif"):
	path_spectral = folder_o/path_camera.name
	
	img_camera   = io.imread(path_camera)
	img_spectral = io.imread(path_spectral)

	from_spectral = measure.regionprops(img_spectral)
	for cell in measure.regionprops(img_camera):
		size_camera.append(cell.area)
		
		x,y = map(lambda x:int(x),cell.centroid)
		idx_spectral = img_spectral[x,y]
		area_spectral = 0 if idx_spectral==0 else from_spectral[idx_spectral-1].area
		size_spectral.append(area_spectral)
# %%
plt.figure()
plt.scatter(
	size_camera,size_spectral,
	s=8,c=[0,0,0,0],edgecolors='k',
	label=path_camera.stem.rpartition("_")[2]
)
plt.plot(
	np.arange(0,max(size_camera)),
	np.arange(0,max(size_camera)),
	'k'
)
plt.title("Segmented Cell Area")
plt.xlabel("Bright Field / px")
plt.ylabel("Confocal / px")
plt.gca().set_aspect('equal')
plt.savefig("plots/cell_segment_comparison.png",dpi=300)

# %%
size_camera   = np.array(size_camera)
size_spectral = np.array(size_spectral)

mod = sm.OLS(size_spectral,size_camera)    # y vs. x
res = mod.fit()
print(res.summary())

#                                  OLS Regression Results                                
# =======================================================================================
# Dep. Variable:                      y   R-squared (uncentered):                   0.952
# Model:                            OLS   Adj. R-squared (uncentered):              0.952
# Method:                 Least Squares   F-statistic:                          1.722e+04
# Date:                Fri, 29 Mar 2024   Prob (F-statistic):                        0.00
# Time:                        18:05:43   Log-Likelihood:                         -4487.8
# No. Observations:                 861   AIC:                                      8978.
# Df Residuals:                     860   BIC:                                      8982.
# Df Model:                           1                                                  
# Covariance Type:            nonrobust                                                  
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1             0.9665      0.007    131.236      0.000       0.952       0.981
# ==============================================================================
# Omnibus:                      460.612   Durbin-Watson:                   1.877
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            33296.579
# Skew:                           1.571   Prob(JB):                         0.00
# Kurtosis:                      33.303   Cond. No.                         1.00
# ==============================================================================

# Notes:
# [1] R² is computed without centering (uncentered) since the model does not contain a constant.
# [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# %%
