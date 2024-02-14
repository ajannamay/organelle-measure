import h5py
import numpy as np
import pandas as pd
from pathlib import Path

folder_i = Path("images/preprocessed")
path_o = Path("data/image_error_uncertainty.csv")

def img_err_multichannel(path_probs,path_error):
    with h5py.File(str(path_probs),'r') as f_probs:
        img_probs = f_probs["exported_data"][:]
    with h5py.File(str(path_error),'r') as f_error:
        img_error = f_error["exported_data"][:]

    img_truth = np.argmax(img_probs,axis=0)

    img_truth_pex = (img_truth==1)
    img_asked_pex = ((2.*img_probs[1]-1.)<img_error[0])
    img_inner_pex = np.logical_and(img_truth_pex,img_asked_pex)
    img_outer_pex = np.logical_and(
                        np.logical_not(img_truth_pex),
                        img_asked_pex
                    )

    img_truth_vac = (img_truth>0)
    img_asked_vac = ((1.-2.*img_probs[0])<img_error[0])
    img_inner_vac = np.logical_and(img_truth_vac,img_asked_vac)
    img_outer_vac = np.logical_and(
                        np.logical_not(img_truth_vac),
                        img_asked_vac
                    )

    return {
        "pixel_lower": [
                        (px_lo_p:=np.count_nonzero(img_inner_pex)),
                        (px_lo_v:=np.count_nonzero(img_inner_vac))
                       ],
        "pixel_upper": [
                        (px_up_p:=np.count_nonzero(img_outer_pex)),
                        (px_up_v:=np.count_nonzero(img_outer_vac))
                       ],
        "error_lower": [
                        px_lo_p/count_p if (count_p:=np.count_nonzero(img_truth_pex))!=0 else 0,
                        px_lo_v/count_v if (count_v:=np.count_nonzero(img_truth_vac))!=0 else 0
                       ],
        "error_upper": [
                        px_up_p/count_p if count_p!=0 else 0,
                        px_up_v/count_v if count_v!=0 else 0
                       ]
    }

def img_err(path_probability,path_uncertainty):
    with h5py.File(str(path_probability),'r') as f_probs:
        img_probs = f_probs["exported_data"][:]
    with h5py.File(str(path_uncertainty),'r') as f_error:
        img_error = f_error["exported_data"][:]

    img_truth = (img_probs[1]>0.5)
    img_asked = (img_probs[1]-img_probs[0]<img_error[0])
    img_inner = np.logical_and(img_truth,img_asked)
    img_outer = np.logical_and(
                    np.logical_not(img_truth),
                    img_asked
                )

    return {
        "pixel_lower": (px_lo:=np.count_nonzero(img_inner)),
        "pixel_upper": (px_up:=np.count_nonzero(img_outer)),
        "error_lower": px_lo/count if (count:=np.count_nonzero(img_truth))!=0 else 0,
        "error_upper": px_up/count if count!=0 else 0
    }


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
    for path_probs in (folder_i/subfolder).glob("probability*.h5"):
        path_error = folder_i/subfolder/path_probs.name.replace("probability","uncertainty")
        organelle = path_probs.stem.partition('_')[2].partition('_')[0]
        if subfolder == "leucine-large-blue-gaussian":
            entry_output = {
                "filename": [f"{subfolder}/{path_probs.stem.partition('_')[2]}"]*2,
                "organelle": ["peroxisome","vacuole"]
            }
            dict_error = img_err_multichannel(path_probs,path_error)
            list_output.append(pd.DataFrame(entry_output|dict_error))
        else:
            entry_output = {
                "filename": f"{subfolder}/{path_probs.stem.partition('_')[2]}",
                "organelle": organelle
            }
            dict_error = img_err(path_probs,path_error)
            list_output.append(pd.DataFrame((entry_output|dict_error),index=[0]))
        print("...",entry_output["filename"])
        # list_output.append({
        #     "path_prob": path_probs,
        #     "path_errs": path_error,
        #     "organelle": organelle
        # })
# df_test = pd.concat(pd.DataFrame(dic,index=[0]) for dic in list_output)
# df_test.to_csv(str(path_o),index=False)
df_errors = pd.concat((df for df in list_output),ignore_index=True)
df_errors.to_csv(str(path_o),index=False)
