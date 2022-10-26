import numpy as np
from pathlib import Path
from skimage import util,io
from organelle_measure import tools


for nd2 in Path("images/Validate6colorWT/nd2/").glob("*.nd2"):
    print(nd2)
    image = tools.load_nd2_plane(str(nd2),frame="cyx",axes="z",idx=0).astype(int)
    io.imsave(f"images/Validate6colorWT/tif/{nd2.stem}.tif",util.img_as_uint(image))

