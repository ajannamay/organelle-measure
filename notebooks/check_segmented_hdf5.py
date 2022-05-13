import h5py
import numpy as np
from pathlib import Path
from skimage import io

path_test = Path("./test")
name_prob = "probability_peroxisome_EYrainbow_glu-100_field-2.h5"
name_errs = "uncertainty_peroxisome_EYrainbow_glu-100_field-2.h5"
name_odsg = "binary-peroxisome_EYrainbow_glu-100_field-2.tiff"

with h5py.File(str(path_test/name_prob),'r') as f_prob:
    probability = f_prob["exported_data"][:]
with h5py.File(str(path_test/name_errs),'r') as f_errs:
    uncertainty = f_errs["exported_data"][:]
difference = np.abs(probability[0]-probability[1])
is_within = (difference < uncertainty[0])

print(np.any(probability[0]==0.5))
# True

print(np.any(probability[1]==0.5))
# True

print(np.all((probability[0]==0.5)==(probability[1]==0.5)))
# True

print(np.all((probability[0]<0.5)==(probability[1]>0.5)))
# True

print(np.all((probability[0]>0.5)==(probability[1]<0.5)))
# True

