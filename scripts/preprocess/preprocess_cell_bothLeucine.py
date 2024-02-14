import pandas as pd
from pathlib import Path
from skimage import io,util,measure,morphology,segmentation
from organelle_measure.tools import batch_apply

def label_cells(path_in,path_out):
    img = io.imread(str(path_in))
    img = measure.label(img)
    img = segmentation.clear_border(img)
    # img = morphology.closing(img)
    for prop in measure.regionprops(img):
        if prop.area<50:
            img[img==prop.label] = 0
    io.imsave(str(path_out),util.img_as_uint(img))
    return None

folders_in = ["EYrainbow_leucine_bin","EYrainbow_leucine_large_bin"]
folders_out = ["EYrainbow_leucine","EYrainbow_leucine_large"]

list_in = []
list_out = []
for i,folder in enumerate(folders_in):
    for file_in in (Path("./data/cell")/folder).glob("*.tif"):
        list_in.append(file_in)
        path_out = Path("./data/cell")/folders_out[i]/file_in.name
        list_out.append(path_out)
args = pd.DataFrame({
    "path_in":list_in,
    "path_out":list_out
})

batch_apply(label_cells,args)