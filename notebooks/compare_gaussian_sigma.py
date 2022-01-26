# This file compares different sigmas for gaussian filter as a preprocessing 
# method. The raw images from the microscope looked better on the controlling 
# software because it used "spline" as a "rescaling" method. 
# The algorithm is not known, that's why we have this file.

import numpy as np
from pathlib import Path
from skimage import util,io,filters
from organelle_measure.tools import load_nd2_plane

list_sigma = [0.3,0.5,0.8,1.0,1.5,2.]

path_img = Path("../test/raw/unmixed-blue-experimental_1nmpp1-3000_field-2.nd2")
img_pex3 = load_nd2_plane(str(path_img),frame="zyx",axes="c",idx=0)
img_vph1 = load_nd2_plane(str(path_img),frame="zyx",axes="c",idx=1)

for sigma in list_sigma:
    out_pex3 = filters.gaussian(img_pex3,sigma=sigma)
    out_vph1 = filters.gaussian(img_vph1,sigma=sigma)
    path_pex3 = Path("../test/gaussian/")/f"pex3_gaussian-{str(sigma).replace('.','-')}_{path_img.stem.partition('_')[2]}.tif"
    path_vph1 = Path("../test/gaussian/")/f"vph1_gaussian-{str(sigma).replace('.','-')}_{path_img.stem.partition('_')[2]}.tif"
    io.imsave(str(path_pex3),util.img_as_float(out_pex3))
    io.imsave(str(path_vph1),util.img_as_float(out_vph1))

# Test RESULT of blue channels:
# For pex3, sigma=1.5 seems too much, while sigma=0.3 is definitely not enough
# sigma=1.0 gives more globular results than sigma=0.8.
# So sigma=1.0 it is. 
# Watershed is necessary.
# For vph1, sigma=1.0 and 0.8 are good. 
# Excluding peroxisome from vacuole images are definitely necessary.

for sigma in list_sigma:
    out_vph1 = filters.difference_of_gaussians(img_vph1,low_sigma=sigma) # default high_sigma=1.6*low_sigma 
    out_vph1 = (out_vph1-(out_min:=out_vph1.min()))/(out_vph1.max()-out_min)
    path_vph1 = Path("../test/difference_of_gaussians/")/f"vph1_diffgaussian_{str(sigma).replace('.','-')}_{path_img.stem.partition('_')[2]}.tif"
    io.imsave(str(path_vph1),util.img_as_float(out_vph1))

for sigma in list_sigma:
    for sigma2 in list_sigma:
        if not sigma2 > sigma:
            continue
        out_vph1 = filters.difference_of_gaussians(img_vph1,low_sigma=sigma,high_sigma=sigma2)
        out_vph1 = (out_vph1-(out_min:=out_vph1.min()))/(out_vph1.max()-out_min)
        path_vph1 = Path("../test/difference_of_gaussians/")/f"vph1_diffgaussian_{str(sigma).replace('.','-')}_{str(sigma2).replace('.','-')}_{path_img.stem.partition('_')[2]}.tif"
        io.imsave(str(path_vph1),util.img_as_float(out_vph1))

# Test RESULT of vph1 channel:
# small low sigmas do not work well.
# low_sigma = 1.5 and 2 are obviously too large.
# low=0.5,high=1.5 is not bad
# low=0.5,high=2.0 is good
# imageJ coordinate (115,302) seems to be upper left corner of a cell with >1 vacuoles.
# low=0.8,high=1.0 and 1.5 are good
# low=1.0,high=2.0 is not bad
# low=1.5,high=2.0 is not good

# Note after Calculation
# The FWHM of 1d gaussian function is 2.355*sigma
# This means at 1.1775*sigma from the origin, the intensity becomes 1/8.
import plotly.express as px
test = np.zeros((9,9,9))
test[4,4,4] = 1
for sig in [0.3,0.5,0.8,1.0,1.5]:
    fig = px.imshow(filters.gaussian(test,sigma=sig)[4],title=f"sigma={sig}")
    fig.show()
# The result shows sigm=0.8 gives the pixels within ball(1) half weight as center,
# while the whole affected region is around ball(2).