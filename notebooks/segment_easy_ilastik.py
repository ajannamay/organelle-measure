# This script is created because there is a modification on my desktop that has 
# not been pushed to remote.
import numpy as np
from skimage import io,util,exposure,filters,segmentation
from organelle_measure.tools import load_nd2_plane

path_pex3 = "../test/raw/unmixed-blue-experimental_1nmpp1-3000_field-2.nd2"
img_pex3 = load_nd2_plane(path_pex3,frame="zyx",axes="c",idx=0)
gauss_max = np.percentile(img_pex3,99.999)
img_norm = exposure.rescale_intensity(
    img_pex3,
    in_range=(img_pex3.min(),img_pex3.max()),
    out_range=(0.,1.)
)
img_gaus = filters.gaussian(img_norm,sigma=0.75)
io.imsave(
    "../test/gaussian/pex3_gaussian-0-75_norm-01_1nmpp1-3000_field-2.tiff",
    img_gaus
)
io.imsave(
    "../test/gaussian/pex3_gaussian-0_1nmpp1-3000_field-2.tiff",
    util.img_as_uint(img_pex3.astype(int))
)
path_pexb = "../test/binary/bin-pex3_gaussian-0-75_1nmpp1-3000_field-2.tiff"


path_sec61 = "../test/raw/"