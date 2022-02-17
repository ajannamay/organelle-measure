import numpy as np
from pathlib import Path
from skimage import io,util
from organelle_measure.tools import load_nd2_plane

folder = "../data/raw/EYrainbow_leucine_large"
output = "../data/leucine-large-cell"
for filename in Path(folder).glob("spectral-green*.nd2"):
    img = load_nd2_plane(str(filename),frame='zyx',axes='t',idx=0).astype(int)
    img = np.mean(img,axis=0,dtype=int)
    io.imsave(
        str(Path(output)/f'{filename.stem.partition("_")[2]}.tif'),
        util.img_as_uint(img)
    )
