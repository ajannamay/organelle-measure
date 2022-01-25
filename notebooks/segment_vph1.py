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

import numpy as np
from skimage import io,util,morphology
from pathlib import Path
# The pipeline is as follows:
# - start with the unmixed-**.nd2 image
# - apply a difference of gaussians filter: vph1_diffgaussian-**.tif
# - use ilastika to binarize the image: bin-vph1-**.tif
# - skeletonize the image: skeleton-vph1-**.tif
# - flood fill the image so that closed ring will be recognized as vacuoles: skeleton-fill-vph1-**.tif

# REPRODUCE THE PROPOSED PIPELINE FROM IMAGEJ TO PYTHON

# Use Scikit-Image to close the segmented images:
path_seg = "../test/binary/bin-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff"
img_seg  = io.imread(path_seg)
img_seg -= img_seg.min()
for rad in [1,2,3]:
    img_open = morphology.binary_closing(img_seg,selem=morphology.ball(rad))
    io.imsave(path_seg.replace("vph1",f"vph1_close-{rad}"),util.img_as_ubyte(img_open))
# ball(3) almost filling every hole so useless. 
# ball(1) so good that I want it solely as output.
# opening potentially gives false positive by connecting borders of several vacuoles.

# Use Scikit-Image to skeletonize
# skeletonize can only take [0,1] as values!
path_seg = "../test/binary/bin-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff"
img_seg  = io.imread(path_seg)
img_seg -= img_seg.min() # ilastik gives results of 1 and 2 for binary segmentation :-(
img_skl  = np.zeros_like(img_seg)
for z in range(len(img_skl)):
    img_skl[z]  = morphology.skeletonize(img_seg[z])
io.imsave(path_seg.replace("/binary","/skeleton").replace("bin","scikit-skeleton2d"),util.img_as_ubyte(img_skl))

# Use Scikit-Image to flood fill the skeletonized
# flood() gives a mask of the region that have same/similar value as the seed.
# flood_fill() gives a img that equals the original plus the result of flood() 
# in our case the flood fill should have smaller blobs than flood()
path_skl = "../test/skeleton/scikit-skeleton2d-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff"
img_skl  = ~io.imread(path_skl).astype(bool) # notice the reverse.
img_fill = np.zeros_like(img_skl)
for z in range(len(img_fill)):
    img_fill[z] = morphology.flood_fill(img_skl[z],(0,0),False,selem=morphology.disk(1))
io.imsave(path_skl.replace("skeleton/scikit-skeleton2d","vacuole/python-fill"),util.img_as_ubyte(img_fill))

# IMPROVEMENT:
# closing the image (ilastik segmentation OR skeletonized)
# with different (2d OR 3d) 
# shapes (disk/ball(1/2/3) OR square/cube(1/3/5))

# Closing the Skletonized Image:
path_skl = "../test/skeleton/scikit-skeleton2d-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff"
img_skl  = io.imread(path_skl).astype(bool)
# 2d:
selem_params_2d = (
    ("disk",  1),
    ("square",3),
    ("disk",  2),
    ("square",5),
    ("disk",  3),
    ("square",7)
)
for sel_shape,sel_rad in selem_params_2d:
    img_clo  = np.zeros_like(img_skl)
    for z in range(len(img_clo)):
        img_clo[z] = morphology.binary_closing(img_skl[z],selem=getattr(morphology,sel_shape)(sel_rad))
    io.imsave(path_skl.replace("scikit",f"close2d-{sel_shape}-{sel_rad}"),img_clo)
# 2d results do not differ too much, 
# but there have been some filled holes in disk(2),
# and square(3) does help with some broken rings.
# I feel that we need to close the images before skeletonization.
# 3d:
path_skl = "../test/skeleton/scikit-skeleton2d-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff"
img_skl  = io.imread(path_skl).astype(bool)
selem_params_3d = (
    ("ball",1),
    ("cube",3),
    ("ball",2),
    ("cube",5),
    ("ball",3),
    ("cube",7)
)
for sel_shape,sel_rad in selem_params_3d:
    img_clo = morphology.binary_closing(img_skl,selem=getattr(morphology,sel_shape)(sel_rad))
    io.imsave(path_skl.replace("scikit",f"close3d-{sel_shape}-{sel_rad}"),img_clo)
# 3d results also not too different.
# cube(5) helps a little, cube(7) is definitely too much wrong.
# One advantage of closing skeletonized images is that it hardly changes the shape of the circle.
# We need to fill the holes to see the results better.

# I think we can conclude that 3d is better than 2d, 
# and ball(2) is not trivial.


# Closing the Segmented Images:
path_seg = "../test/binary/bin-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff"
img_seg  = io.imread(path_seg)
img_seg  = (img_seg - img_seg.min()).astype(bool) # ilastik gives results of 1 and 2 for binary segmentation :-(
# 2d:
selem_params_2d = (
    ("disk",  1),
    ("square",3),
    ("disk",  2),
    ("square",5),
    ("disk",  3),
    ("square",7)
)
for sel_shape,sel_rad in selem_params_2d:
    img_clo  = np.zeros_like(img_skl)
    for z in range(len(img_clo)):
        img_clo[z] = morphology.binary_closing(img_skl[z],selem=getattr(morphology,sel_shape)(sel_rad))
    io.imsave(path_seg.replace("bin-",f"close2d-{sel_shape}-{sel_rad}-"),img_clo)
# 3d:
selem_params_3d = (
    ("ball",1),
    ("cube",3),
    ("ball",2),
    ("cube",5),
    ("ball",3),
    ("cube",7)
)
for sel_shape,sel_rad in selem_params_3d:
    img_clo = morphology.binary_closing(img_skl,selem=getattr(morphology,sel_shape)(sel_rad))
    io.imsave(path_seg.replace("bin-",f"close3d-{sel_shape}-{sel_rad}-"),img_clo)

# Skeletonize the closed images:
from pathlib import Path
for path_close in Path("../test/binary").glob("close*.tiff"):
    img_bin = io.imread(str(path_close)).astype(bool)
    img_skl = np.zeros_like(img_bin)
    for z in range(len(img_skl)):
        img_skl[z] = morphology.skeletonize(img_bin[z])
    io.imsave(str(Path("../test/skeleton")/f"skeleton2d-{path_close.name}"),img_skl)

# Flood Fill All the Skeletonized Images
for path_skeleton in Path("../test/skeleton").glob("*skeleton2d*.tiff"):
    img_skl = ~io.imread(str(path_skeleton)).astype(bool)
    img_fil = np.zeros_like(img_skl)
    for z in range(len(img_skl)):
        img_fil[z] = morphology.flood_fill(img_skl[z],(0,0),False,selem=morphology.disk(1))
    io.imsave(str(Path("../test/vacuole")/f"python-fill-{path_skeleton.name}"),img_fil)
