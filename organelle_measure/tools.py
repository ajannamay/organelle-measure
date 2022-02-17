import numpy as np
import pandas as pd
from skimage import morphology,measure

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
# END of ND2reader wrapper.

def open_golgi(path):
    if 'c' in get_nd2_size(path):
        return np.mean(load_nd2_plane(str(path),frame="czyx",axes='t',idx=0).astype(int),axis=0,dtype=int)
    else:
        return load_nd2_plane(str(path),frame="zyx",axes='t',idx=0).astype(int)
open_organelles = {
    "peroxisome":   lambda x: load_nd2_plane(str(x),frame="zyx",axes='tc',idx=0).astype(int),
    "vacuole":      lambda x: load_nd2_plane(str(x),frame="zyx",axes='tc',idx=1).astype(int),
    "ER":           lambda x: load_nd2_plane(str(x),frame="zyx",axes='t', idx=0).astype(int),
    "golgi":        open_golgi,
    "mitochondria": lambda x: load_nd2_plane(str(x),frame="zyx",axes='tc',idx=0).astype(int),
    "LD": lambda x: load_nd2_plane(str(x),frame="zyx",axes='tc',idx=1).astype(int)
}

def skeletonize_zbyz(image):
    """image has values [0,1] only."""
    skeletonized = np.zeros_like(image)
    for z in range(len(skeletonized)):
        skeletonized[z] = morphology.skeletonize(image[z])
    return skeletonized

def neighbor_mean(img_orga,img_cell):
    """
    fill the regions with cells with the mean of neighbouring background.
    """
    properties = measure.regionprops(img_cell)
    img_out = np.copy(img_orga)
    for prop in properties:
        min_row, min_col, max_row, max_col = prop.bbox
        win_row,win_col = max_row - min_row, max_col - min_col
        min_row = max(min_row - win_row, 0)
        min_col = max(min_col - win_col, 0)
        max_row = min(max_row + win_row, img_cell.shape[0])
        max_col = min(max_col + win_col, img_cell.shape[1])

        bbox_bin = img_cell[min_row:max_row,min_col:max_col]
        for z in range(img_orga.shape[0]):
            bbox_orga = img_orga[z,min_row:max_row,min_col:max_col]
            samples = bbox_orga[bbox_bin==0]
            background = int(samples.mean())
            idx = np.array([[z,*coo] for coo in prop.coords])
            img_out[tuple(idx.T)] = background
    return img_out

def batch_apply(func,args:pd.DataFrame):
    results = []
    errmsgs  = []
    for entry in args.iterrows():
        try:
            func(**entry[1])
            results.append(True)
            errmsgs.append("")
        except Exception as e:
            results.append(False)
            errmsgs.append(repr(e))
    args["RESULT"] = pd.Series(results)
    args["ERR_MESSAGES"] = pd.Series(errmsgs)
    return None

