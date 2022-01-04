# This file compares different sigmas for gaussian filter as a preprocessing 
# method. The raw images from the microscope looked better on the controlling 
# software because it used "spline" as a "rescaling" method. The algorithm is 
# not known, that's why we have this file.

import numpy as np
from pathlib import Path
from skimage import util,io,filters
from organelle_measure.util import load_nd2_plane

list_sigma = [0.3,0.5,0.8,1.0,1.5,2]

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
