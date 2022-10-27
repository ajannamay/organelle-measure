import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import util,io,morphology
from organelle_measure import tools


for nd2 in Path("data/Validate6colorWT/nd2/").glob("*.nd2"):
    print(nd2,nd2.stem)
    image = tools.load_nd2_plane(str(nd2),frame="cyx",axes="t",idx=0).astype(int)
    io.imsave(f"data/Validate6colorWT/tif/{nd2.stem.replace(' ','_')}.tif",util.img_as_uint(image))

img_er1 = io.imread("data/Validate6colorWT/tif/488_ER.tif")
img_er2 = io.imread("data/Validate6colorWT/tif/488_ER_2.tif")
img_gg1 = io.imread("data/Validate6colorWT/tif/514_golgi.tif")
img_gg2 = io.imread("data/Validate6colorWT/tif/514_golgi_2.tif")

mask_er1 = io.imread("data/Validate6colorWT/segment/Simple Segmentation_488_ER.tif")
mask_er2 = io.imread("data/Validate6colorWT/segment/Simple Segmentation_488_ER_2.tif")
mask_gg1 = io.imread("data/Validate6colorWT/segment/Simple Segmentation_514_golgi.tif")
mask_gg2 = io.imread("data/Validate6colorWT/segment/Simple Segmentation_514_golgi_2.tif")

mask_er1 = (mask_er1==2)
mask_er2 = (mask_er2==2)
mask_gg1 = (mask_gg1==2)
mask_gg2 = (mask_gg2==2)

mask_er1 = morphology.binary_erosion(mask_er1)
mask_er2 = morphology.binary_erosion(mask_er2)
mask_gg1 = morphology.binary_erosion(mask_gg1)
mask_gg2 = morphology.binary_erosion(mask_gg2)

spectra_er1_er = np.transpose(img_er1[:,mask_er1])
spectra_er1_gg = np.transpose(img_er1[:,mask_gg1])

spectra_gg1_er = np.transpose(img_gg1[:,mask_er1])
spectra_gg1_gg = np.transpose(img_gg1[:,mask_gg1])

spectrum_er1_er = np.mean(spectra_er1_er,axis=0)
spectrum_er1_gg = np.mean(spectra_er1_gg,axis=0)

spectrum_gg1_er = np.mean(spectra_gg1_er,axis=0)
spectrum_gg1_gg = np.mean(spectra_gg1_gg,axis=0)


