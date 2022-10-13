import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def img_err(path_probability,n_channel=2):
    """assuming channel 0 is the background."""
    with h5py.File(str(path_probability),'r') as f_probs:
        img_probs = f_probs["exported_data"][:]

    answer = {
            "pixel_lower": np.zeros(n_channel-1),
            "pixel_upper": np.zeros(n_channel-1),
            "error_lower": np.zeros(n_channel-1),
            "error_upper": np.zeros(n_channel-1)
        }
    for a in range(1,n_channel):
        img_truth = (img_probs[a]>(1./n_channel))
        if np.all(np.logical_not(img_truth)):
            continue
        possible_probs = np.unique(img_probs[a])
        threshold = np.min(img_probs[a,img_truth])
        (idx,) = np.nonzero(possible_probs==threshold) # `,` unwraps np.array
        prob_inner = possible_probs[idx[0]+1]
        prob_outer = possible_probs[idx[0]-1]
        img_inner = (img_probs[a]==prob_inner)
        img_outer = (img_probs[a]==prob_outer)
        
        answer["pixel_lower"][a-1] = (px_lo:=np.count_nonzero(img_inner))
        answer["pixel_upper"][a-1] = (px_up:=np.count_nonzero(img_outer))
        answer["error_lower"][a-1] = px_lo/count if (count:=np.count_nonzero(img_truth))!=0 else 0
        answer["error_upper"][a-1] = px_up/count if count!=0 else 0
    return answer


subfolders = [
    "EYrainbow_glucose",
    "EYrainbow_glucose_largerBF",
    "EYrainbow_rapamycin_1stTry",
    "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st",
    "EYrainbow_leucine_large",
    "EYrainbow_leucine",
    "EYrainbowWhi5Up_betaEstrodiol",
    "leucine-large-blue-gaussian"
]

list_output = []
for subfolder in subfolders:
    for path_probs in (Path("images/preprocessed")/subfolder).glob("probability*.h5"):
        organelle = ["peroxisome","vacuole"] if subfolder == "leucine-large-blue-gaussian" else path_probs.stem.partition('_')[2].partition('_')[0]
        channels = 3 if subfolder == "leucine-large-blue-gaussian" else 2
        entry_output = {
            "experiment": subfolder,
            "filename":   path_probs.stem.partition('_')[2],
            "organelle":  organelle
        }
        dict_error = img_err(path_probs,n_channel=channels)
        list_output.append(pd.DataFrame((entry_output|dict_error),index=list(range(channels-1))))
        print("...",entry_output["experiment"],entry_output["filename"])
df_errors = pd.concat((df for df in list_output),ignore_index=True)
df_errors.to_csv("data/image_error_probability_new.csv",index=False)
