# This script is created because there is a modification on my desktop that has 
# not been pushed to remote.
import numpy as np
from pathlib import Path
from skimage import io,util,exposure,filters,segmentation
from organelle_measure.tools import get_nd2_size, load_nd2_plane

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
    io.imsave(
        str(Path("../test/gaussian/")/f'sec61-norm01_{path_green.stem.partition("_")[2]}.tif'),
        img_norm  
    )

    # img_gaus = filters.gaussian(img_norm,sigma=0.75)
    # io.imsave(
    #     str(Path("../test/gaussian/")/f'sec61-gaus075-norm01_{path_green.stem.partition("_")[2]}.tif'),
    #     img_gaus
    # )

    # img_gaus = filters.gaussian(img_norm,sigma=0.25)
    # io.imsave(
    #     str(Path("../test/gaussian/")/f'sec61-gaus025-norm01_{path_green.stem.partition("_")[2]}.tif'),
    #     img_gaus
    # )

# yellow channel
path_yellow_nc = "../test/raw/spectral-yellow_EYrainbow_glu-100_field-2.nd2"
path_yellow_1c = "../test/raw/spectral-yellow_EYrainbowWhi5Up_betaEstrodiol-0_overnight_field-2.nd2"
path_yellow = path_yellow_nc
if "c" in get_nd2_size(path_yellow):
    img = np.sum(load_nd2_plane(path_yellow,frame="czyx",axes="t",idx=0),axis=0)
else:
    img = load_nd2_plane(path_yellow,frame="zyx",axes="t",idx=0)
img_raw = np.array(img).astype(np.uint16)
io.imsave(
    str(Path("../test/gaussian")/f"sec7-raw_{Path(path_yellow).stem.partition('_')[2]}.tif"),
    util.img_as_uint(img_raw)
)
img_norm = exposure.rescale_intensity(img_raw,out_range=(0.,1.))
io.imsave(
    str(Path("../test/gaussian")/f"sec7-norm01_{Path(path_yellow).stem.partition('_')[2]}.tif"),
    img_norm
)
for sig in [0.25,0.50,0.75]:
    img_gaus = filters.gaussian(img_norm,sigma=sig)
    io.imsave(
        str(Path("../test/gaussian")/f"sec7-gaussian-{str(sig).replace('.','-')}_{Path(path_yellow).stem.partition('_')[2]}.tif"),
        img_gaus
    )

# red channels
path_red = "../test/raw/unmixed-red-experimental_1nmpp1-3000_field-2.nd2"
img_tom70 = load_nd2_plane(path_red,frame="zyx",axes="c",idx=0)
img_erg6  = load_nd2_plane(path_red,frame="zyx",axes="c",idx=1)

img_raw_tom70 = np.array(img_tom70).astype(np.uint16)
img_raw_erg6  = np.array(img_erg6).astype(np.uint16)
io.imsave(
    str(Path("../test/gaussian")/f"tom70-raw_{Path(path_red).stem.partition('_')[2]}.tif"),
    util.img_as_uint(img_raw_tom70)
)
io.imsave(
    str(Path("../test/gaussian")/f"erg6-raw_{Path(path_red).stem.partition('_')[2]}.tif"),
    util.img_as_uint(img_raw_erg6)
)

img_norm_tom70 = exposure.rescale_intensity(img_raw_tom70,out_range=(0.,1.))
img_norm_erg6  = exposure.rescale_intensity(img_raw_erg6,out_range=(0.,1.))
io.imsave(
    str(Path("../test/gaussian")/f"tom70-norm01_{Path(path_red).stem.partition('_')[2]}.tif"),
    img_norm_tom70
)
io.imsave(
    str(Path("../test/gaussian")/f"erg6-norm01_{Path(path_red).stem.partition('_')[2]}.tif"),
    img_norm_erg6
)
for sig in [0.50,0.75,1.00]:
    img_gaus_tom70 = filters.gaussian(img_norm_tom70,sigma=sig)
    img_gaus_erg6  = filters.gaussian(img_norm_erg6,sigma=sig)
    io.imsave(
        str(Path("../test/gaussian")/f"tom70-gaussian-{str(sig).replace('.','-')}_{Path(path_red).stem.partition('_')[2]}.tif"),
        img_gaus_tom70
    )
    io.imsave(
        str(Path("../test/gaussian")/f"erg6-gaussian-{str(sig).replace('.','-')}_{Path(path_red).stem.partition('_')[2]}.tif"),
        img_gaus_erg6
    )