# Test Sample: 1nmpp1-3000_field-2

import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from skimage import io,util,exposure,filters,morphology,segmentation,feature,measure
from organelle_measure.tools import load_nd2_plane

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

# END of the results reported in the group meeting.

# After segmenting peroxisomes, I realised between thresholding with histogram 
# and ilastik, there are other ways such as random_walk(). 
# So I am trying that.

# Try More Sigmas of Gaussian
path_raw = "../test/raw/unmixed-blue-experimental_1nmpp1-3000_field-2.nd2"
img_raws = load_nd2_plane(path_raw,frame="zyx",axes="c",idx=1)
img_gaus = filters.gaussian(img_raws,sigma=0.75)
epsilon   = (100.*5)/(512*512)
gauss_max,gauss_min = np.percentile(img_gaus,[100-epsilon,epsilon])
img_norm = exposure.rescale_intensity(img_gaus,in_range=(gauss_min,gauss_max),out_range=(0.,1.))
io.imsave(
    path_raw.replace(
        "unmixed-blue-experimental",
        "vph1_gaussian-0-75"
    ).replace(
        ".nd2",".tif"
    ).replace(
        "raw/","gaussian/"
    ),
    img_norm
)
# By checking the kernel, I chose sigma=0.75. See `./compare_gaussian_sigma.py`

# Get Rid of Peroxisome from Vacuole Label Images
path_gau = "../test/gaussian/vph1_gaussian-0-75_1nmpp1-3000_field-2.tif"
img_gaus = io.imread(path_gau)
path_ref = "../test/peroxisome/watershed_pex3_gaussian-0-75_1nmpp1-3000_field-2.tif"
img_refr = io.imread(path_ref)
img_clen = np.copy(img_gaus)
img_clen[img_refr>0] = 0
io.imsave(
    path_gau.replace("vph1","cleaned-vph1"),
    img_clen
)
# result is shit.

# Get Rid of Peroxisome from Vacule Gaussian Images
path_pex3 = "../test/gaussian/pex3_gaussian-0-75_1nmpp1-3000_field-2.tif"
path_vph1 = "../test/gaussian/vph1_gaussian-0-75_1nmpp1-3000_field-2.tif"
img_pex3 = io.imread(path_pex3)
img_vph1 = io.imread(path_vph1)
img_clen = img_vph1 * (1-img_pex3)
io.imsave(
    path_vph1.replace("vph1","multiply-vph1"),
    img_clen
)
# result is shit again.

# Check the Percentile and Intensities:
import plotly.express as px
percentiles = np.linspace(99.,100.,num=21)
intensities = np.percentile(img_vph1,percentiles)
fig = px.line(x=percentiles,y=intensities)
fig.show()
# super flat at first and a sudden peak near 1. 

# Segmentation by random walk didn't continue because I want to have something 
# to show Shankar. Therefore, I turned back to ilastik.

# Use Watershed on Ilastik Skeleton
path_bin = "../test/skeleton/skeleton-vph1_diffgaussian_0-75_1nmpp1-3000_field-2-1.tif"
img_biny = io.imread(path_bin).astype(bool)
img_dist = np.zeros_like(img_biny,dtype=float)
img_wtsd = np.zeros_like(img_biny,dtype=int)
for z in range(len(img_biny)):
    img_dist[z] = ndi.distance_transform_edt(img_biny[z])
    img_wtsd[z] = segmentation.watershed(-img_dist[z],mask=img_biny[z])
io.imsave(
    "../test/vacuole/watershedzbyz-vph1_diffgaussian_0-75_1nmpp1-3000_field-2.tif",
    util.img_as_uint(img_wtsd)
)
# The result is unexpectedly amazingly good!

def intersection_over_union(bool1,bool2):
    return (bool1*bool2)/(bool1+bool2)

path_fill = "../test/vacuole/"
path_wtsd = "../test/vacuole/"
img_fill = io.imread(path_fill)
img_wtsd = io.imread(path_wtsd)
img_core = measure.label(img_fill)
num_z = len(img_core)
img_xpnd = np.copy(img_core)
for prop in measure.regionprops(img_core):
    
    if prop.area<3:
        img_xpnd[img_core==prop.label] = 0
        continue
    
    z_coords = prop.coords[:,0]
    z_max,z_min = z_coords.max(),z_coords.min()
    
    ref_xpnd = img_xpnd[z_min] # change to img_core?
    for z in range(z_min-1,-1,-1):
        ref_wtsd = img_wtsd[z]
        mask_last = (ref_xpnd==prop.label)
        count_last = len(ref_xpnd[mask_last])
        sample = ref_wtsd[mask_last]
        list_gray = []
        list_IoU  = []
        for gray in np.unique(sample):
            mask_this = (ref_wtsd==gray)
            count_this = len(ref_wtsd[mask_this])
            if count_this < (3*count_last):
                list_gray.append(gray)
                list_IoU.append(intersection_over_union(mask_this,mask_last))
        if len(list_IoU) == 0:
            break
        chosen = list_gray[np.argmax(list_IoU)]
        img_xpnd[z,ref_wtsd==chosen] = prop.label
        ref_xpnd = img_xpnd[z]
        
    ref_xpnd = img_xpnd[z_max] # change to img_core?
    for z in range(z_max+1,num_z):
        ref_wtsd = img_wtsd[z]
        mask_last = (ref_xpnd==prop.label)
        count_last = len(ref_xpnd[mask_last])
        sample = ref_wtsd[mask_last]
        list_gray = []
        list_IoU  = []
        for gray in np.unique(sample):
            mask_this = (ref_wtsd==gray)
            count_this = len(ref_wtsd[mask_this])
            if count_this < (3*count_last):
                list_gray.append(gray)
                list_IoU.append(intersection_over_union(mask_this,mask_last))
        if len(list_IoU) == 0:
            break
        chosen = list_gray[np.argmax(list_IoU)]
        img_xpnd[z,ref_wtsd==chosen] = prop.label
        ref_xpnd = img_xpnd[z]


