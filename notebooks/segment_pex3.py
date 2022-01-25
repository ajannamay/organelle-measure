# SEGMENT PEX3

# peroxiome should be pretty easy to segment, since their signals are bright, 
# and their shape is determined to be globular.

# In Preprocessing, just to make sure that the preprocessed images are properly 
# normalized so that `ilastik` can work consistently over different images.

# In Postprocessing, use `watershed()` to divide connecting peroxisomes, which 
# is important for exclusion of vph1 false postives.

import numpy as np
from skimage import util,io,segmentation,filters
from organelle_measure.util import load_nd2_plane

# Gaussian
path_pex3 = "../test/raw/unmixed-blue-experimental_1nmpp1-3000_field-2.nd2"
raws_pex3 = load_nd2_plane(path_pex3,frame="zyx",axes="c",idx=0)
for radius in np.linspace(0.6,1.5,10,endpoint=True):
    gaus_pex3 = filters.gaussian(raws_pex3,sigma=0.8)
    epsilon   = 1./(gaus_pex3.shape[0]*1000*1000)
    gauss_max,gauss_min = np.percentile(gaus_pex3,[100-epsilon,epsilon])
    norm_pex3 = (gaus_pex3-gauss_min)/(gauss_max-gauss_min)
    norm_pex3[norm_pex3>1.] = 1.0
    norm_pex3[norm_pex3<0.] = 0.
    io.imsave(path_pex3.replace("unmixed-blue-experimental",f"pex3_gaussian-{str(radius).replace('.','-')}").replace(".nd2",".tif"),norm_pex3)

# Watershed, not tested yet.

