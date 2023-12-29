# %%
import numpy as np
import pandas as pd
from pathlib import Path
from nd2reader import ND2Reader


# We're trying to prove the red channels separate 
# mitochondria and lipid droplets correctly
# by using a strain that has a different color for LDs(?)
folder_img = "images/raw/Kiandohkt4colorWT"

# %% ND2READER DOES NOT WORK FOR IMAGES >7 CHANNELS
for file_img in Path(folder_img).glob("*.nd2"):
	with ND2Reader(str(file_img)) as img:
		print(file_img,img.sizes)


# %%
import javabridge
import bioformats

javabridge.start_vm(class_path=bioformats.JARS)

# %%
path_img = Path(folder_img)/"EY2796_4color_FOV1_WT_bfp.nd2"
with ND2Reader(str(path_img)) as nd2_img:
	size_img = nd2_img.sizes
	
sample_img  = bioformats.load_image(
				str(path_img), 
				c=None, z=0, t=0, series=None, index=None,
				rescale=False, wants_max_intensity=False, 
				channel_names=None
              )
array_img = np.empty((size_img['z'],*sample_img.shape))
for z in range(size_img['z']):
	array_img[z] = bioformats.load_image(
						str(path_img), 
						c=None, z=z, t=0, series=None, index=None,
						rescale=False, wants_max_intensity=False, 
						channel_names=None
            	   )


# %%
javabridge.kill_vm()

# %%
