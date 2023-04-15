import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = "data/condensate"

px_x,px_y,px_z = 0.1083333, 0.1083333, 0.2
voxel = px_x * px_y * px_z
pixel = px_x * px_y

indices = [0,1,2,6,7,8,9]

meta = {
    "vacuole": {
        "folder":    "WT CFP",
        "prefix":    "WT",
        "zeros":     "",
        "ax_suffix": "axis"
    },
    "condensate": {
        "folder":    "WT-YFP-Data-11.7",
        "prefix":    "YFP_WT",
        "zeros":     "00",
        "ax_suffix": "cell_stats"
    }
}

buffer_cell_vacuole    = []
buffer_cell_condensate = []
buffer_organelle = []
for idx in indices:
    for organelle in ["vacuole","condensate"]:
        array_axis = np.loadtxt(f"{folder}/{meta[organelle]['folder']}/axis/{meta[organelle]['prefix']}{meta[organelle]['zeros']}{idx}{meta[organelle]['ax_suffix']}.csv",delimiter=',')
        array_cell = np.loadtxt(f"{folder}/{meta[organelle]['folder']}/cells/{meta[organelle]['prefix']}{meta[organelle]['zeros']}{idx}cellSize.csv",delimiter=',')
        matrix_vol = np.loadtxt(f"{folder}/{meta[organelle]['folder']}/volumes/{meta[organelle]['prefix']}{meta[organelle]['zeros']}{idx}volumes.csv",delimiter=',')
        if organelle == "vacuole":
            buffer_cell_vacuole.append(pd.DataFrame({
                    "fov":            idx,
                    "cell_idx":       list(range(len(array_cell))),
                    "cell_major":     array_axis[0],
                    "cell_minor":     array_axis[1],
                    "cell_area":      array_cell,
            }))
        if organelle == "condensate":
            buffer_cell_condensate.append(pd.DataFrame({
                    "fov":            idx,
                    "cell_idx":       list(range(len(array_cell))),
                    "cell_major":     array_axis[0],
                    "cell_minor":     array_axis[1],
                    "cell_area":      array_cell,
            }))
        for j in range(len(array_cell)):
            array_vol = matrix_vol[j]
            not0_vol = []
            if organelle == "vacuole":
                not0_vol = list(filter(lambda x: x > 0., np.array(array_vol)))
            if organelle == "condensate":
                not0_vol = list(filter(lambda x: not np.isnan(x), np.array(array_vol)))
            buffer_organelle.append(pd.DataFrame({
                "fov":            idx,
                "cell_idx":       j,
                "organelle_name": organelle,
                "organelle_idx":  list(range(len(not0_vol))),
                "organelle_vol":  not0_vol
            }))
df_cell_vacuole    = pd.concat(buffer_cell_vacuole,   ignore_index=True)
df_cell_condensate = pd.concat(buffer_cell_condensate,ignore_index=True)
df_organelle = pd.concat(buffer_organelle,ignore_index=True)

assert df_cell_vacuole.equals(df_cell_condensate), "inconsistent cell properties from different organelles data sources."
df_cell = df_cell_vacuole
df_cell = df_cell.copy()
df_cell = df_cell.loc[df_cell["cell_area"].gt(200),:]

df_cell.set_index(["fov","cell_idx"],inplace=True)

df_cell.loc[:,"cell_volume"] = 2 * (np.sqrt(df_cell.loc[:,"cell_area"] * pixel))**3/np.sqrt(np.pi)
df_cell.loc[:,"vacuole_volume"] = df_organelle.loc[df_organelle["organelle_name"].eq("vacuole"),:].groupby(["fov","cell_idx"]).sum()["organelle_vol"]
df_cell.loc[:,"condensate_volume"] = df_organelle.loc[df_organelle["organelle_name"].eq("condensate"),:].groupby(["fov","cell_idx"]).sum()["organelle_vol"] * voxel

df_cell["vacuole_volume"].fillna(0.,inplace=True)
df_cell["condensate_volume"].fillna(0.,inplace=True)
df_cell.dropna(inplace=True)

df_cell["vacuole_fraction"]    = df_cell["vacuole_volume"]/df_cell["cell_volume"]
df_cell["condensate_fraction"] = df_cell["condensate_volume"]/df_cell["cell_volume"]


plt.scatter(
    x=df_cell.loc[:,"cell_volume"],
    y=df_cell.loc[:,"condensate_fraction"],
    alpha=0.5, edgecolors='w'
)
plt.scatter(
    x=df_cell.loc[:,"cell_volume"],
    y=df_cell.loc[:,"vacuole_fraction"],
    alpha=0.5, edgecolors='w'
)
plt.scatter(
    x=df_cell.loc[:,"cell_volume"],
    y=df_cell.loc[:,"vacuole_fraction"] + df_cell.loc[:,"condensate_fraction"],
    alpha=0.5, edgecolors='w', label=r'$+$'
)
plt.scatter(
    x=df_cell.loc[:,"cell_volume"],
    y=df_cell.loc[:,"vacuole_fraction"] - df_cell.loc[:,"condensate_fraction"],
    alpha=0.5, edgecolors='w', label=r'$-$'
)
plt.legend()