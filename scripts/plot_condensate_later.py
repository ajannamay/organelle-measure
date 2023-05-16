from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/condensate/Fig4data.csv",header=[0,1])

plt.rcParams['font.size'] = 24
fig,ax = plt.subplots(figsize=(14,9))
ax.errorbar(
    x=df.loc[:,("vacuole vol frac no condensate","cell size")],
    y=df.loc[:,("vacuole vol frac no condensate","phi_vac - phi_vac_0")],
    yerr=df.loc[:,("vacuole vol frac no condensate","s.e.m.")],
    ls='none',ecolor="grey",capsize=8,alpha=0.75,
    marker='o',mfc="tab:blue",mec='tab:blue',ms=15,
    label=r"$\phi^{vacuole}_{no\ condensate}$"
)
ax.errorbar(
    x=df.loc[:,("vacuole vol frac with condensate","cell size")],
    y=df.loc[:,("vacuole vol frac with condensate","phi_vac - phi_vac_0")],
    yerr=df.loc[:,("vacuole vol frac with condensate","s.e.m.")],
    ls='none',ecolor="grey",capsize=8,alpha=0.75,
    marker='o',mfc="tab:green",mec='tab:green',ms=15,
    label=r"$\phi^{vacuole}_{with\ condensate}$"
)
ax.plot(
    df.loc[:,("prediction","cell size")],
    df.loc[:,("prediction","phi_vac - phi_vac_0")],
    c="black",linestyle='--',alpha=0.75,
    label="prediction"
)
ax.errorbar(
    x=[],
    y=[],
    yerr=[],
    ls='none',ecolor="grey",capsize=8,alpha=0.75,
    marker='o',mfc="tab:red",mec='tab:red',ms=15,
    label=r"$\phi_{condensate}$"
)
ax.legend(bbox_to_anchor=(1.02, 1),fontsize=28)
ax.set_xlabel(r"Cell Size / $\mu m^3$",fontsize=28)
ax.set_ylabel(r"$\phi-\phi_0$",fontsize=28)

# inlet
inlet = fig.add_axes([0.20,0.60,0.30,0.25])
inlet.errorbar(
    x=df.loc[:,("condensate vol frac","cell size")],
    y=df.loc[:,("condensate vol frac","phi_condensate")],
    yerr=df.loc[:,("condensate vol frac","s.e.m.")],
    ls='none',ecolor="grey",capsize=5,alpha=0.75,
    marker='o',mfc="tab:red",mec='tab:red',ms=10,
    label=r"$\phi_{condensate}$"
)
inlet.set_ybound(-0.01,0.09)
inlet.set_xticks(np.arange(40,120,10))
inlet.set_yticks(np.arange(0,0.10,0.02))
inlet.tick_params(axis='x', labelsize=16)
inlet.tick_params(axis='y', labelsize=18)
fig.savefig("./data/condensate/figure-4-Vcell.png",bbox_inches='tight')



# Use V_cell's z-score as x axis.

plt.rcParams['font.size'] = 24
fig,ax = plt.subplots(figsize=(14,9))
ax.errorbar(
    x=(df.loc[:,("vacuole vol frac no condensate","cell size")] - df.loc[:,("vacuole vol frac no condensate","cell size")].mean())/df.loc[:,("vacuole vol frac no condensate","cell size")].mean(),
    y=df.loc[:,("vacuole vol frac no condensate","phi_vac - phi_vac_0")],
    yerr=df.loc[:,("vacuole vol frac no condensate","s.e.m.")],
    ls='none',ecolor="grey",capsize=8,alpha=0.75,
    marker='o',mfc="tab:blue",mec='tab:blue',ms=15,
    label=r"$\phi^{vacuole}_{no\ condensate}$"
)
ax.errorbar(
    x=(df.loc[:,("vacuole vol frac with condensate","cell size")] - df.loc[:,("vacuole vol frac with condensate","cell size")].mean())/df.loc[:,("vacuole vol frac with condensate","cell size")].mean(),
    y=df.loc[:,("vacuole vol frac with condensate","phi_vac - phi_vac_0")],
    yerr=df.loc[:,("vacuole vol frac with condensate","s.e.m.")],
    ls='none',ecolor="grey",capsize=8,alpha=0.75,
    marker='o',mfc="tab:green",mec='tab:green',ms=15,
    label=r"$\phi^{vacuole}_{with\ condensate}$"
)
ax.plot(
    (df.loc[:,("prediction","cell size")] - df.loc[:,("prediction","cell size")].mean())/df.loc[:,("prediction","cell size")].mean(),
    df.loc[:,("prediction","phi_vac - phi_vac_0")],
    c="black",linestyle='--',alpha=0.75,
    label="prediction"
)
ax.errorbar(
    x=[],
    y=[],
    yerr=[],
    ls='none',ecolor="grey",capsize=8,alpha=0.75,
    marker='o',mfc="tab:red",mec='tab:red',ms=15,
    label=r"$\phi_{condensate}$"
)
ax.legend(bbox_to_anchor=(1.02, 1),fontsize=28)
ax.set_xlabel(r"$(V_{cell} - \left<V_{cell}\right>)\ /\ \left<V_{cell}\right>$",fontsize=28)
ax.set_ylabel(r"$\phi-\phi_0$",fontsize=28)

# inlet
inlet = fig.add_axes([0.20,0.60,0.30,0.25])
inlet.errorbar(
    x=(df.loc[:,("condensate vol frac","cell size")] - df.loc[:,("condensate vol frac","cell size")].mean())/df.loc[:,("condensate vol frac","cell size")].mean(),
    y=df.loc[:,("condensate vol frac","phi_condensate")],
    yerr=df.loc[:,("condensate vol frac","s.e.m.")],
    ls='none',ecolor="grey",capsize=5,alpha=0.75,
    marker='o',mfc="tab:red",mec='tab:red',ms=10,
    label=r"$\phi_{condensate}$"
)
inlet.set_ybound(-0.01,0.09)
inlet.set_yticks(np.arange(0,0.10,0.02))
inlet.tick_params(axis='x', labelsize=16)
inlet.tick_params(axis='y', labelsize=18)
fig.savefig("./data/condensate/figure-4-Vnormed.png",bbox_inches='tight')



