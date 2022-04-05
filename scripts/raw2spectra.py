# This script prints out the spectrum of the raw images,
# in order to validate our unmixing.

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from skimage import io
from organelle_measure.tools import load_nd2_plane

folder_i = Path("test")
folder_o = Path("data/spectra")

file_label1 = folder_i/"label-peroxisome_EYrainbow_glu-100_field-0.tiff"
file_label2 = folder_i/"label-LD_EYrainbow_glu-100_field-0.tiff"
file_raw    = folder_i/"spectral-red_EYrainbow_glu-100_field-0.nd2"

img_label1 = io.imread(str(file_label1))
img_label2 = io.imread(str(file_label2))
img_raw    = load_nd2_plane(str(file_raw),frame="zcyx",axes="t",idx=0)


