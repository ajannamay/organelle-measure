# The image shown on Nikon Ti2 softwares are preprocessed with "spline" 
# rescaling, and that is why the images look good on the software while not so 
# good on ImageJ and python. So in this notebook I am going to try 2 ways to do
# the spline:
# 1. use univariate spline on x and y direction, and then do the average.
# 2. use bivariate spline immediately on the whole image.
# The second is ideal, but I am afraid the speed would be too slow.
# 
# In another project I tried to do similar things because of the aurora-like 
# noise in red channels. The project did work out well so I used neighbourhood 
# mean as the solution for that project.

import numpy as np
from organelle_measure import util as orzutil
from pathlib import Path
from scipy import interpolate
from skimage import util, io

# io
path_raw = "../test/raw/unmixed-blue-experimental_1nmpp1-0_field-3.nd2"
img_raw = orzutil.load_nd2_plane(path_raw,frame="zyx",axes='c',idx=1).astype(int)[18]
dir_out = "../test/spline/"


# ============================================================================ #
# interpolate twice in 1d and take average 
# refactored into spline1d2()
x,y = img_raw.shape

arr_x = np.arange(x)
fun_1dx = interpolate.interp1d(arr_x,img_raw,kind="cubic",axis=0)
out_x = np.linspace(0,x-1,x*4)
img_1dx = fun_1dx(out_x)

arr_y = np.arange(y)
fun_1dy = interpolate.interp1d(arr_y,img_1dx,kind="cubic",axis=1)
out_y = np.linspace(0,y-1,y*4)
img_1dy = fun_1dy(out_y)

# io.imshow(img_1dx.astype(int)==img_1dy.astype(int))
io.imsave(f"{dir_out}/interpolate-{Path(path_raw).stem}.tif",util.img_as_uint(img_1dy.astype(np.uint16)))

# the result is really bad. Most interpolated values are near zero, and become
# 65535 after img_as_uint(().astype(np.uint16)). Besides bad interpolatition, 
# the type is also a problem.


# ============================================================================ #
# interpolate once in 2d
# refactored into spline2d()
