import numpy as np
from skimage import color,io,util,morphology


organelles = [
    "ER",
    "vacuole",
    "mitochondria",
    "LD",
    "golgi",
    "peroxisome",
]
colors = {
    "peroxisome":   [0.0,0.0,1.0], # blue
    "vacuole":      [0.0,1.0,1.0], # cyan
    "ER":           [0.0,1.0,0.0], # green
    "golgi":        [1.0,1.0,0.0], # yellow
    "mitochondria": [1.0,0.5,0.0], # organge
    "LD":           [1.0,0.0,0.0]  # red
}

folder = "../data/preprocessed/EYrainbow_glucose_largerBF"
img_stk = np.zeros((512,512,3))
for organelle in organelles:
    fpaths = f"binary-{organelle}_EYrainbow_glu-200_field-4.tiff"
    img_raw = io.imread(f"{folder}/{fpaths}")[18]
    img_bin = util.img_as_float(img_raw>1)
    img_clr = color.gray2rgb(img_bin)
    img_out = img_clr*colors[organelle]
    io.imsave(f"z18-{fpaths}",img_out)
#     img_ske = morphology.skeletonize(img_bin)
#     img_ske = morphology.binary_closing(img_ske,selem=morphology.disk(1))
#     if organelle in ["peroxisome","golgi","LD"]:
#         img_ske = morphology.binary_closing(img_ske,selem=morphology.disk(2))
#     img_ske_clr = color.gray2rgb(img_ske)
#     img_stk += (img_ske_clr*colors[organelle])
# io.imsave(f"stack-EYrainbow_glu-200_field-4.tiff",img_stk)

