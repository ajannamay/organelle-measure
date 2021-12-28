import numpy as np
from pathlib import Path

# BEGIN of ND2reader wrapper:
from nd2reader import ND2Reader

def get_nd2_size(path:str):
    with ND2Reader(path) as images:
        size = images.sizes
    return size

def load_nd2_plane(path:str,frame:str='cyx',axes:str='tz',idx:int=0):
    """read an image flexibly with ND2Reader."""
    with ND2Reader(path) as images:
        images.bundle_axes = frame
        images.iter_axes = axes
        img = images[idx]
    return img.squeeze()

dict_open_organelles = {
    "pex3": lambda x: load_nd2_plane(str(x),frame="zyx",axes='tc',idx=0),
    "vph1": lambda x: load_nd2_plane(str(x),frame="zyx",axes='tc',idx=1),
    "sec61":lambda x: load_nd2_plane(str(x),frame="zyx",axes='t', idx=0),
    "sec7": lambda x: np.sum(load_nd2_plane(str(x),frame="czyx",axes='t',idx=0),axis=0),
    "tom70":lambda x: load_nd2_plane(str(x),frame="zyx",axes='tc',idx=0),
    "erg6": lambda x: load_nd2_plane(str(x),frame="zyx",axes='tc',idx=1)
}
# END of ND2reader wrapper.