# SEGMENT PEX3

# peroxiome should be pretty easy to segment, since their signals are bright, 
# and their shape is determined to be globular.

# In Preprocessing, just to make sure that the preprocessed images are properly 
# normalized so that `ilastik` can work consistently over different images.

# In Postprocessing, use `watershed()` to divide connecting peroxisomes, which 
# is important for exclusion of vph1 false postives.

import numpy as np
from skimage import io,filters,exposure,feature,segmentation,util,measure,morphology,color
from organelle_measure.tools import load_nd2_plane

# Gaussian
# maybe we should normalize when reading the nd2 file, and then do the gaussian 
# without normalization other than truncating values > 1.
# because usually the nd2 pex3 signals are over satuated so the maximum is 
# determined.
path_raws = "../test/raw/unmixed-blue-experimental_1nmpp1-3000_field-2.nd2"
img_raws = load_nd2_plane(path_raws,frame="zyx",axes="c",idx=0)
img_gaus = filters.gaussian(img_raws,sigma=0.75)
epsilon  = (100.*5)/(512*512) # 5 is a parameter
gauss_max,gauss_min = np.percentile(img_gaus,[100-epsilon,epsilon])
img_norm = exposure.rescale_intensity(img_gaus,in_range=(gauss_min,gauss_max),out_range=(0.,1.))
io.imsave(
    path_raws.replace(
        "unmixed-blue-experimental",
        f"pex3_gaussian-0-75"
    ).replace(
        ".nd2",".tif"
    ).replace(
        "raw/","gaussian/"
    ),
    img_norm
)
# the difference is easily wiped out by different LUTs.
# but the result of sigma=1.0 seems more round than 0.8

# A New Idea That Does Not Require Ilastik: 
# use the gaussian image as input,
# calculate local maximums as counters and centroids of peroxisomes
# calculate the region around them that are brighter than half of the maxima 
# half of the local maximum
path_gaus = "../test/gaussian/pex3_gaussian-0-75_1nmpp1-3000_field-2.tif"
img_gaus = io.imread(path_gaus)
# threshold = 5.*np.median(img_gaus)
threshold = 0.01
idx_maxm = feature.peak_local_max(img_gaus,min_distance=3,threshold_abs=0.02)

# I'm super satisfied with this result!
# But we need to check the universility of its effectiveness.
img_maxm = np.zeros_like(img_gaus,dtype=bool)
img_maxm[tuple(idx_maxm.T)] = True
img_maxm = measure.label(img_maxm)
img_thre = (img_gaus>threshold)
img_wtsd = segmentation.watershed(-img_gaus,markers=img_maxm,mask=img_thre)
io.imsave(
    path_gaus.replace(
        "gaussian/","peroxisome/watershed_"
    ),
    util.img_as_uint(img_wtsd)
)

# # flood fill the bright region around local maxima
# img_thre = np.zeros_like(img_gaus,dtype=np.int16)
# img_half = (img_gaus>(0.01))
# for i,coords in enumerate(idx_maxm):
#     img_fill = segmentation.flood(img_half,tuple(coords))
#     img_thre[img_fill] = i+1
# io.imsave(
#     path_gaus.replace(
#         "gaussian/","peroxisome/connected_"
#     ),
#     util.img_as_uint(img_thre)
# )

# showing the result of peaking finding
img_maxm = np.zeros_like(img_gaus,dtype=bool)
img_maxm[tuple(idx_maxm.T)] = True
img_maxm = measure.label(img_maxm)
img_maxm = morphology.dilation(img_maxm,selem=morphology.ball(1))
io.imsave(
    path_gaus.replace(
        "gaussian/","maxima/"
    ).replace(
        "pex3","label-pex3"
    ),
    util.img_as_uint(img_maxm)
)
