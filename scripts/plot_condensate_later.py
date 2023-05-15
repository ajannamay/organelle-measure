from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/condensate/Fig4data.csv",header=[0,1])

plt.rcParams['font.size'] = 24
fig,ax = plt.subplots(figsize=(15,9))
ax.errorbar(
    x=df.loc[:,("vacuole vol frac no condensate","cell size")],
    y=df.loc[:,("vacuole vol frac no condensate","phi_vac - phi_vac_0")],
    yerr=df.loc[:,("vacuole vol frac no condensate","s.e.m.")],
    ls='none',ecolor="grey",capsize=3,alpha=0.75,
    marker='o',markerfacecolor="tab:blue",markeredgecolor='tab:blue',
    label=r"$\phi^{vacuole}_{no\ condensate}$"
)
ax.errorbar(
    x=df.loc[:,("vacuole vol frac with condensate","cell size")],
    y=df.loc[:,("vacuole vol frac with condensate","phi_vac - phi_vac_0")],
    yerr=df.loc[:,("vacuole vol frac with condensate","s.e.m.")],
    ls='none',ecolor="grey",capsize=3,alpha=0.75,
    marker='o',markerfacecolor="tab:green",markeredgecolor='tab:green',
    label=r"$\phi^{vacuole}_{with\ condensate}$"
)
ax.scatter(
    x=df.loc[:,("prediction","cell size")],
    y=df.loc[:,("prediction","phi_vac - phi_vac_0")],
    c="tab:orange",edgecolors='tab:orange',alpha=0.75,
    label="prediction"
)
ax.errorbar(
    x=[],
    y=[],
    yerr=[],
    ls='none',ecolor="grey",capsize=3,alpha=0.75,
    marker='o',markerfacecolor="tab:red",markeredgecolor='tab:red',
    label=r"$\phi_{condensate}$"
)
ax.legend(bbox_to_anchor=(1.02, 1))

# inlet
inlet = fig.add_axes([0.18,0.60,0.30,0.25])
inlet.errorbar(
    x=df.loc[:,("condensate vol frac","cell size")],
    y=df.loc[:,("condensate vol frac","phi_condensate")],
    yerr=df.loc[:,("condensate vol frac","s.e.m.")],
    ls='none',ecolor="grey",capsize=3,alpha=0.75,
    marker='o',markerfacecolor="tab:red",markeredgecolor='tab:red',
    label=r"$\phi_{condensate}$"
)
inlet.set_ybound(-0.01,0.09)
