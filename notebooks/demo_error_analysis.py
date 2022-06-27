import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from pathlib import Path
from skimage import io

path_i = Path("images/preprocessed/EYrainbow_glucose_largerBF/probability_mitochondria_EYrainbow_glu-100_field-5.h5")

with h5py.File(str(path_i),"r") as rfile:
    img = rfile["exported_data"][:]

np.unique(img[1])

io.imshow(img[1,16])
demo = px.imshow(img[1,16,180:230,50:100])
demo.write_html("demo_mito.html")

df_errors = pd.read_csv("data/image_error_probability.csv")
df_errors["experiment"] = list(map(lambda x:x.partition("/")[0],df_errors["filename"].to_list()))
df_average = df_errors.groupby(["organelle","experiment"]).mean()
df_average.reset_index(inplace=True)
sns.barplot(data=df_average,x="organelle",y="error_lower",hue="experiment",ci=None)
sns.barplot(data=df_average,x="organelle",y="error_upper",hue="experiment",ci=None)

