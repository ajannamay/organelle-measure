import numpy as np
import pandas as pd

# for path in Path("data/condensate").glob("**/*.csv"):
#     array = np.loadtxt(str(path),delimiter=',')
#     print(array.shape,path)

folder = "data/condensate"

px_x,px_y,px_z = 0.1083333, 0.1083333, 0.2

indices = [0,1,2,6,7,8,9]

meta = {
    "vacuole": {
        "prefix":    "WT",
        "folder":    "WT CFP",
        "zeros":     "",
        "ax_suffix": "axis"
    },
    "condensate": {
        "prefix":    "YFP_WT",
        "folder":    "WT-YFP-Data-11.7",
        "zeros":     "00",
        "ax_suffix": "cell_stats"
    }
}

buffer_cell = []
buffer_organelle = []
for idx in indices:
    for organelle in ["vacuole","condensate"]:
        array_axis = np.loadtxt(f"{folder}/{meta[organelle]['folder']}/axis/{meta[organelle]['prefix']}{meta[organelle]['zeros']}{idx}{meta[organelle]['ax_suffix']}.csv",delimiter=',')
        array_cell = np.loadtxt(f"{folder}/{meta[organelle]['folder']}/cells/{meta[organelle]['prefix']}{meta[organelle]['zeros']}{idx}cellSize.csv",delimiter=',')
        matrix_vol = np.loadtxt(f"{folder}/{meta[organelle]['folder']}/volumes/{meta[organelle]['prefix']}{meta[organelle]['zeros']}{idx}volumes.csv",delimiter=',')
        buffer_cell.append(pd.DataFrame({
                "fov":            idx,
                "cell_idx":       list(range(len(array_cell))),
                "cell_major":     array_axis[0],
                "cell_minor":     array_axis[1],
                "cell_area":      array_cell,
        }))
        for j in range(len(array_cell)):
            array_vol = matrix_vol[j]
            not0_vol = list(filter(lambda x: not np.isnan(x), np.array(array_vol)))
            buffer_organelle.append(pd.DataFrame({
                "fov":            idx,
                "cell_idx":       j,
                "organelle_name": organelle,
                "organelle_idx":  list(range(len(not0_vol))),
                "organelle_vol":  not0_vol
            }))
df_cell      = pd.concat(buffer_cell,     ignore_index=True)
df_organelle = pd.concat(buffer_organelle,ignore_index=True)
