# SEGMENT PEX3

# peroxiome should be pretty easy to segment, since their signals are bright, 
# and their shape is determined to be globular.

# In Preprocessing, just to make sure that the preprocessed images are properly 
# normalized so that `ilastik` can work consistently over different images.

# In Postprocessing, use `watershed()` to divide connecting peroxisomes, which 
# is important for exclusion of vph1 false postives.

import numpy as np
from skimage import io,filters,feature,segmentation,util,measure,morphology,color
from organelle_measure.tools import load_nd2_plane

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
    io.imsave(
        path_pex3.replace(
            "unmixed-blue-experimental",
            f"pex3_gaussian-{str(radius).replace('.','-')}"
        ).replace(
            ".nd2",".tif"
        ).replace(
            "raw/","gaussian/"
        ),
        norm_pex3
    )
# I don't see any observable difference from sigma=0.6 to 1.5.

# A New Idea That Does Not Require Ilastik: 
# use the gaussian image as input,
# calculate local maximums as counters and centroids of peroxisomes
# calculate the region around them that are brighter than half of the maxima 
# half of the local maximum
path_gaus = "../test/gaussian/pex3_gaussian-0-8_1nmpp1-3000_field-2.tif"
img_gaus = io.imread(path_gaus)
idx_maxm = feature.peak_local_max(img_gaus,min_distance=2,threshold_abs=0.03)

# I'm super satisfied with this result!
# But we need to check the universility of its effectiveness.
img_maxm = np.zeros_like(img_gaus,dtype=bool)
img_maxm[tuple(idx_maxm.T)] = True
img_maxm = measure.label(img_maxm)
img_thre = (img_gaus>0.01)
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

# # showing the result of peaking finding
# img_maxm = np.zeros_like(img_gaus,dtype=bool)
# img_maxm[tuple(idx_maxm.T)] = True
# img_maxm = measure.label(img_maxm)
# img_maxm = morphology.dilation(img_maxm,selem=morphology.ball(1))
# img_maxm = color.gray2rgb(img_maxm)
# io.imsave(
#     path_gaus.replace(
#         "gaussian/","maxima/"
#     ).replace(
#         "pex3","label-pex3"
#     ),
#     img_maxm
# )


# Watershed, not tested yet.

