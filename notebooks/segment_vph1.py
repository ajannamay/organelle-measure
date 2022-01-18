# Test Sample: 1nmpp1-3000_field-2

# Difficulties are:
# - peroxisomes are showing in the unmixed vacuole channel as dot false positives. This is very common.
# - in some planes vacuole signals do not form a closed ring.
# - the image quality

# In order to find the volumeric vacuoles, some time ago I tried the first half and propose the following:
# 1. using the ilastik to do the segmentation of vph1 signals
# 2. using imageJ/python to skeletonize the binary images
# 3. using python binary 3D dilation/closing for
# 4. using python to flood fill 2D the closed ring structures
# 5. (optional) assuming there can be only one vacuole accross different z, and fit with a cube.

# Ilastik Result: different sigmas for the gaussian filter:
# - bkgd_vph1-vph1_gaussian03_1nmpp1TriK_field2.ilp: sigma=0.3 gives too many salt and smoke noises in ilastik. Also this proj does not save 
# - bkgd_vph1-vph1_gaussian05_1nmpp1TriK_field2.ilp: sigma=0.5 don't like it very much
# - bkgd_vph1-vph1_gaussian08_1nmpp1TriK_field2.ilp: sigma=0.8 like the result very much
# - bkgd_vph1-vph1_gaussian1_1nmpp1TriK_field2.ilp: do not see so many rings by eyes, actually none :-(
# - bkgd_vph1-vph1_gaussian15_1nmpp1TriK_field2.ilp: sigma=1.5 very bad, the hole in the center is almost invisible

# Ilastik Result: the sigma_low and sigma_high of DoG filters:
# - small low sigmas do not work well.
# - low_sigma = 1.5 and 2 are obviously too large.
# - low=0.5,high=1.5 is not bad
# - low=0.5,high=2.0 is good
# - imageJ coordinate (115,302) seems to be upper left corner of a cell with >1 vacuoles.
# - low=0.8,high=1.0 and 1.5 are good
# - low=1.0,high=2.0 is not bad
# - low=1.5,high=2.0 is not good

# Use Scikit-Image to skeletonize:
import numpy as np
from skimage import io,util,morphology
from skimage.util.dtype import img_as_bool, img_as_ubyte
path_seg = "../test/binary/bin-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff"
img_seg  = io.imread(path_seg)
img_seg -= img_seg.min()
for rad in [1,2,3]:
    img_open = morphology.binary_closing(img_seg,selem=morphology.ball(rad))
    io.imsave(path_seg.replace("vph1",f"vph1_open-{rad}"),util.img_as_ubyte(img_open))
# ball(3) almost filling every hole so useless. 
# ball(1) so good that I want it solely as output.
# opening potentially gives false positive by connecting borders of several vacuoles.

# Use Scikit-Image to skeletonize
# skeletonize can only take [0,1] as values!
path_seg = "../test/binary/bin-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff"
img_seg  = io.imread(path_seg)
img_seg -= img_seg.min()
img_skl  = np.zeros_like(img_seg)
for z in range(len(img_skl)):
    img_skl[z]  = morphology.skeletonize(img_seg[z])
io.imsave(path_seg.replace("/binary","/skeleton").replace("bin","scikit-skeleton2d"),util.img_as_ubyte(img_skl))

# Use Scikit-Image to flood fill the skeletonized
path_seg = "../test/binary/bin-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff"
img_seg  = io.imread(path_seg)



# The pipeline is as follows:
# - start with the unmixed-**.nd2 image
# - apply a difference of gaussians filter: vph1_diffgaussian-**.tif
# - use ilastika to binarize the image: bin-vph1-**.tif
# - skeletonize the image: skeleton-vph1-**.tif
# - flood fill the image so that closed ring will be recognized as vacuoles: skeleton-fill-vph1-**.tif

# A New Possible Pipeline:
# Since 3d skeletonize is do-able, we can use the old way:
# XOR(skeletonized,convexhull(skeleonized))
# but we need to check those small false positives.