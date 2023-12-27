import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from organelle_measure.data import read_results

# Global Variables
plt.rcParams["figure.autolayout"]=True
plt.rcParams['font.size'] = '26'
px_x,px_y,px_z = 0.41,0.41,0.20

organelles = [
    "peroxisome",
    "vacuole",
    "ER",
    "golgi",
    "mitochondria",
    "LD"
]

df_bycell = read_results(Path("./data/results"),["paperRebuttal"],(px_x,px_y,px_z))
