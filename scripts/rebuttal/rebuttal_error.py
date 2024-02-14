import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# =========================================================================
organelles = {
    "peroxisome":    "peroxisome",
    "vacuole":       "vacuole",
    "ER":            "ER",
    "Golgi":         "golgi",
    "mitochondrion": "mitochondria",
    "lipid droplet": "LD"
}

folders = [
    "EYrainbow_glucose_largerBF",
    "EYrainbow_leucine_large",
    "EYrainbowWhi5Up_betaEstrodiol",
    "EYrainbow_1nmpp1_1st",
    "EYrainbow_rapamycin_1stTry"
]

def parse_meta_organelle(name):
    """name is the stem of the ORGANELLE label image file."""
    organelle = name.partition("_")[2].partition("_")[0]
    if "1nmpp1" in name:
        experiment = "1nmpp1"
    elif "betaEstrodiol" in name:
        experiment = "betaEstrodiol"
    else:
        experiment = name.partition("EYrainbow_")[2].partition("-")[0]
    condition = name.partition(f"{experiment}-")[2].partition("_")[0]
    field = name.partition("field-")[2]    
    return {
        "experiment": experiment,
        "condition":  condition,
        "hour":       3,
        "field":      field,
        "organelle":  organelle
    }

def segment_at_thresholds(filepath):
    prob_file = h5py.File(str(filepath),"r")
    prob_data = prob_file["exported_data"]

    N = 100
    scanned = np.zeros(N)
    for j,threshold in enumerate(np.linspace(0,1,N,endpoint=False)):
        scanned[j] = np.count_nonzero((prob_data[1]>threshold))
    return scanned

dfs_segment = []
count = 0
for fd in folders:
    # print(f"Starting processing {fd}...")
    for file in Path(f"images/preprocessed/{fd}").glob("probability_*.h5"):
        count += 1
        # meta = parse_meta_organelle(file.stem)
        # meta["segmented"] = segment_at_thresholds(file)
        # dfs_segment.append(meta)
        # print(f"....{file} scanned.")


df_segment = pd.concat([pd.DataFrame(df) for df in dfs_segment])

for df in dfs_segment:
    np.savetxt(
        f'data/rebuttal_error/byfov/{df["organelle"]}_{df["experiment"]}_{df["condition"]}_field-{df["field"]}',
        df["segmented"][0],
        fmt="%i"
    )


# =========================================================================

dfs_segment = []
for path in Path("data/rebuttal_error/").glob("*"):
    organelle ,_,rest  = path.stem.partition("_")
    experiment,_,rest  = rest.partition("_")
    condition ,_,rest  = rest.partition("_")
    rest      ,_,field = rest.partition("field-")
    segments = np.loadtxt(str(path))
    dfs_segment.append(pd.DataFrame({
        "organelle":  organelle,
        "experiment": experiment,
        "condition":  condition,
        "field":      field,
        "segmented": [segments]
    }))
df_segment = pd.concat(dfs_segment)

grouped_histo = df_segment[["organelle","experiment","segmented"]].groupby(["organelle","experiment"]).sum()
grouped_histo["pixels_mean"] = grouped_histo.apply(lambda x: np.mean(x["segmented"]),axis=1)
grouped_histo["pixels_std"] = grouped_histo.apply(lambda x: np.std(x["segmented"]),axis=1)
grouped_histo["traditional"] = grouped_histo.apply(lambda x: x["segmented"][50],axis=1)
