import h5py
import numpy as np
import pandas as pd
from pathlib import Path

folder_i = Path("")
fodler_o = Path("")

path_truth = Path("test/probability_peroxisome_EYrainbow_glu-100_field-2.h5")
path_error = Path("test/uncertainty_peroxisome_EYrainbow_glu-100_field-2.h5")

with h5py.File(str(path_truth),'r') as f_probs:
    img_probs = f_probs["exported_data"][:]
with h5py.File(str(path_error),'r') as f_error:
    img_error = f_error["exported_data"][:]

img_truth = (img_probs[1]>0.5)
img_asked = (img_probs[1]-img_probs[0]>img_error[0])
img_inner = np.logical_and(img_truth,img_asked)
img_outer = np.logical_and(
                np.logical_not(img_truth),
                img_asked
            )
