import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology,measure,segmentation

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

# START of organelle nd2 file opener
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
# END of organelle nd2 file opener

# START of vacuole postprocesser

def skeletonize_zbyz(binary3d):
    """image has values [0,1] only."""
    skeletonized = np.zeros_like(binary3d)
    for z in range(len(skeletonized)):
        skeletonized[z] = morphology.skeletonize(binary3d[z])
    return skeletonized
def watershed_zbyz(skeleton3d):
    reversed3d = ~skeleton3d
    img_dist = np.zeros_like(reversed3d,dtype=float)
    img_wtsd = np.zeros_like(reversed3d,dtype=int)
    for z in range(len(reversed3d)):
        img_dist[z] = ndi.distance_transform_edt(reversed3d[z])
        img_wtsd[z] = segmentation.watershed(-img_dist[z],mask=reversed3d[z],)
    return img_wtsd
def find_complete_rings(skeleton3d):
    core3d = np.zeros_like(skeleton3d,dtype=int)
    for z in range(core3d.shape[0]):
        skeleton3d[z] = segmentation.clear_border(skeleton3d[z])
        core3d[z] = measure.label(morphology.flood_fill(~skeleton3d[z],(0,0),False,selem=morphology.disk(1)))
    return core3d
def intersection_over_union(bool1,bool2):
    return np.count_nonzero(bool1*bool2)/np.count_nonzero(bool1+bool2)
def find_hidden_object(expand2d,watershed2d,coord):
    """
    expand2d: 2d integer image with the objects on previous plane.
    watershed: 2d watershed image on the current plane
    label: label of the object of interest
    """
    mask_last = (expand2d==expand2d[coord])
    label = watershed2d[coord]
    mask_this = (watershed2d==label)
    iou = intersection_over_union(mask_this,mask_last)
    if iou<0.5 or (np.count_nonzero(mask_this)>1.5*np.count_nonzero(mask_last)):
        return None
    else:
        return label
def better_vacuole_img(filled3d,watershed3d):
    core3d = measure.label(filled3d)
    num_z = core3d.shape[0]
    expanded3d = np.copy(core3d)
    for prop in measure.regionprops(core3d):
        # exclude very small blobs
        if prop.area<20:
            expanded3d[core3d==prop.label] = 0
            continue
        # find the range on z axis
        z_coords = prop.coords[:,0]
        z_max,z_min = z_coords.max(),z_coords.min()
        centroid = tuple(int(coo) for coo in prop.centroid[1:])
        
        # expand towards -z direction
        ref_xpnd = expanded3d[z_min]
        for z in range(z_min-1,-1,-1):
            ref_wtsd = watershed3d[z]
            chosen = find_hidden_object(ref_xpnd,ref_wtsd,centroid)
            if chosen is None:
                break
            expanded3d[z,ref_wtsd==chosen] = prop.label
            ref_xpnd = expanded3d[z]
        # expand towards +z direction
        ref_xpnd = expanded3d[z_max] 
        for z in range(z_max+1,num_z):
            ref_wtsd = watershed3d[z]
            chosen = find_hidden_object(ref_xpnd,ref_wtsd,centroid)
            if chosen is None:
                break
            expanded3d[z,ref_wtsd==chosen] = prop.label
            ref_xpnd = expanded3d[z]
    return expanded3d
# END of vacuole postprocesser


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

