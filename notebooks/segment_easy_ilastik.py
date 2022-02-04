# This script is created because there is a modification on my desktop that has 
# not been pushed to remote.
from json import load
import numpy as np
from pathlib import Path
from skimage import io,util,exposure,filters,segmentation
from organelle_measure.tools import load_nd2_plane

path_pex3 = "../test/raw/unmixed-blue-experimental_1nmpp1-3000_field-2.nd2"
img_pex3 = load_nd2_plane(path_pex3,frame="zyx",axes="c",idx=0)
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

for path_blue in Path("../test/raw/").glob("*blue-experimental*.nd2"):
    img = load_nd2_plane(str(path_blue),frame="zyx",axes="c",idx=0)
    io.imsave(
        str(Path("../test/gaussian/")/f'pex3-raw-int_{path_blue.stem.partition("_")[2]}.tif'),
        util.img_as_uint(img.astype(int))
    )

    img_gaus = exposure.rescale_intensity(img,in_range='image',out_range=(0.,1))
    img_gaus = filters.gaussian(img_gaus,sigma=0.75)
    io.imsave(
        str(Path("../test/gaussian/")/f'pex3-gaus075-norm01_{path_blue.stem.partition("_")[2]}.tif'),
        img_gaus
    )

    epsilon  = (100.*1.)/(512*512) # 1 is a parameter
    gauss_max,gauss_min = np.percentile(img_gaus,[100-epsilon,epsilon])
    img_norm = exposure.rescale_intensity(img_gaus,in_range=(gauss_min,gauss_max),out_range=(0.,1.))
    img_norm = filters.gaussian(img_norm,sigma=0.75)
    io.imsave(
        str(Path("../test/gaussian/")/f'pex3-gaus075-normeps_{path_blue.stem.partition("_")[2]}.tif'),
        img_norm
    )

for path_green in Path("../test/raw/").glob("*green*.nd2"):
    img = load_nd2_plane(str(path_green),frame="zyx",axes="t",idx=0)
    # io.imsave(
    #     str(Path("../test/gaussian/")/f'sec61-raw-int_{path_green.stem.partition("_")[2]}.tif'),
    #     util.img_as_uint(img.astype(int))
    # )
    img_norm = exposure.rescale_intensity(img,in_range="image",out_range=(0.,1.))
    # img_gaus = filters.gaussian(img_norm,sigma=0.75)
    # io.imsave(
    #     str(Path("../test/gaussian/")/f'sec61-gaus075-norm01_{path_green.stem.partition("_")[2]}.tif'),
    #     img_gaus
    # )
    img_gaus = filters.gaussian(img_norm,sigma=0.25)
    io.imsave(
        str(Path("../test/gaussian/")/f'sec61-gaus025-norm01_{path_green.stem.partition("_")[2]}.tif'),
        img_gaus
    )
    