import scipy
import numpy as np
import pandas as pd
from pathlib import Path

for path in Path("data/condensate").glob("**/*.csv"):
    array = np.loadtxt(str(path),delimiter=',')
    print(array.shape,path)

px_x,px_y,px_z = 0.1083333, 0.1083333, 0.2


