import numpy as np
from skimage import util,io

img = io.imread("../test/probability/prob-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff")
img_out = img[:,0,...]<img[:,1,...]
io.imsave("../test/binary/bin-vph1_diffgaussian_0-8_1nmpp1-3000_field-2.tiff",util.img_as_ubyte(img_out))

# Using probability as output of ilastik does not make an improvement. 
# The output segmentation is just the binary check of (prob1<prob2)