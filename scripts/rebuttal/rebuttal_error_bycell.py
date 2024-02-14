# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from batch_apply import batch_apply
from skimage import io,measure # type: ignore
import h5py

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

# CALCULATE OVER DATA & SAVE INTO FILES
# %% HELPING FUNCTIONS
def parse_meta_organelle(name):
    """name is the stem of the ORGANELLE label image file."""
    organelle = name.partition("_")[2].partition("_")[0]
    if "1nmpp1" in name:
        experiment = "1nmpp1"
    elif "betaEstrodiol" in name:
        experiment = "betaEstrodiol"
    elif "spectral-blue" in name: # from EYrainbow-leucine-large
        experiment = "leu"

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
def segment_at_thresholds(image):
    N = 100
    scanned = np.zeros(N)
    for j,threshold in enumerate(np.linspace(0,1,N,endpoint=False)):
        scanned[j] = np.count_nonzero((np.round(image,2)>threshold))
    return scanned
# END HELPING FUNCTIONS

# %% CALCULATE ALL EXPERIMENTS EXCEPT LEUCINE LARGE
def calculate_error_bycell(path_orga,path_cell,folder_out):
    img_cell  = io.imread(str(path_cell))
    prob_file = h5py.File(str(path_orga),"r")
    prob_data = prob_file["exported_data"]
    img_orga  = prob_data[1] # type: ignore

    meta = parse_meta_organelle(path_orga.stem)
    results = []
    record = {}
    for cell in measure.regionprops(img_cell):
        record["cell_idx"] = cell.label
        min_row, min_col, max_row, max_col = cell.bbox
        img_cell_crop = cell.image
        img_orga_crop = img_orga[:,min_row:max_row,min_col:max_col] # type: ignore
        for z in range(img_cell_crop.shape[0]):
            img_orga_crop[z] = img_orga_crop[z] * img_cell_crop # type: ignore
        scanned = segment_at_thresholds(img_orga_crop)
        for sc in range(len(scanned)):
            record[f"scanned-{sc}"] = [int(scanned[sc])]
        # record["traditional"] = scanned[50]
        # record["mean"]        = np.mean(scanned)
        # record["std"]         = np.std(scanned)
        # record["error"]       = record["std"]/10
        # record["pct_err"]     = record["error"]/record["mean"]
        results.append(pd.DataFrame(meta | record))
    df_result = pd.concat(results)
    df_result.to_csv(
        f"{folder_out}/{path_orga.stem.partition('probability_')[2]}.csv",
        index=False
    )
    print(f"...finished {path_orga}")
    return None


list_cells = []
list_organelles = []
# for folder in folders:
#     for path_organelle in Path(f"images/preprocessed/{folder}").glob("probability*.h5"):
#         list_organelles.append(path_organelle)
#         path_cell = Path(f"images/cell/{folder}/binCell_{path_organelle.stem.partition('probability_')[2].partition('_')[2]}.tif")
#         list_cells.append(path_cell)
for folder in folders:
    for path_cell in Path(f"images/cell/{folder}").glob("binCell*.tif"):
        for organelle in organelles.keys():
            list_cells.append(path_cell)
            path_organelle = Path(f"images/preprocessed/{folder}/probability_{organelles[organelle]}_{path_cell.stem.partition('_')[2]}.h5")
            list_organelles.append(path_organelle)

args = pd.DataFrame({
    "path_cell":  list_cells,
    "path_orga":  list_organelles,
    "folder_out": "data/rebuttal_error/bycell_nonzero/"
})
batch_apply(calculate_error_bycell,args)

# def test_exists(path_orga,path_cell,folder_out):
#     for path in [path_orga,path_cell,folder_out]:
#         if not Path(path).exists():
#             raise ValueError(f"Path not exist: {path}")
#     return None
# batch_apply(test_exists,args)
# >>> Leucine Large Experiment has special peroxisome and vacuole images.

# %% Leucine Large Experiment
# =======================================================================
# Peroxisomes are pixels whose largest intensities are     on channel[1].
# Vacuoles    are pixels whose largest intensities are not on channel[0].

def calculate_error_blues(path_orga,path_cell,folder_out):    
    img_cell = io.imread(str(path_cell))
    prob_file = h5py.File(str(path_orga),"r")
    prob_data = prob_file["exported_data"]
    img_peroxisome = prob_data[1] # type: ignore
    img_vacuole = prob_data[1] + prob_data[2] # type: ignore

    meta = parse_meta_organelle(path_orga.stem)
    results_peroxisome = []
    results_vacuole    = []
    record_peroxisome  = {}
    record_vacuole     = {}
    for cell in measure.regionprops(img_cell):
        record_peroxisome["cell_idx"] = cell.label
        record_peroxisome["organelle"] = "peroxisome"

        record_vacuole["cell_idx"] = cell.label
        record_vacuole["organelle"] = "vacuole"
        
        min_row, min_col, max_row, max_col = cell.bbox
        img_cell_crop = cell.image

        img_pex_crop = img_peroxisome[:,min_row:max_row,min_col:max_col] # type: ignore
        for z in range(img_pex_crop.shape[0]): # type: ignore
            img_pex_crop[z] = img_pex_crop[z] * img_cell_crop # type: ignore
        scanned_peroxisome = segment_at_thresholds(img_pex_crop)
        for sc in range(len(scanned_peroxisome)):
            record_peroxisome[f"scanned-{sc}"] = [int(scanned_peroxisome[sc])]
        results_peroxisome.append(pd.DataFrame(meta | record_peroxisome))

        img_vac_crop = img_vacuole[:,min_row:max_row,min_col:max_col] # type: ignore
        for z in range(img_vac_crop.shape[0]):
            img_vac_crop[z] = img_vac_crop[z] * img_cell_crop # type: ignore
        scanned_vacuole = segment_at_thresholds(img_vac_crop)
        for sc in range(len(scanned_vacuole)):
            record_vacuole[f"scanned-{sc}"] = [int(scanned_vacuole[sc])]
        results_vacuole.append(pd.DataFrame(meta | record_vacuole))

    df_peroxisome = pd.concat(results_peroxisome)
    df_vacuole    = pd.concat(results_vacuole)

    df_peroxisome.to_csv(
        f"{folder_out}/peroxisome_{path_orga.stem.partition('spectral-blue_')[2]}.csv",
        index=False
    )
    df_vacuole.to_csv(
        f"{folder_out}/vacuole_{path_orga.stem.partition('spectral-blue_')[2]}.csv",
        index=False
    )
    return None

list_cells = []
list_blues = []
for path_cell in Path(f"images/cell/EYrainbow_leucine_large/").glob("binCell*.tif"):
    list_cells.append(path_cell)
    path_blue = Path(f"images/preprocessed/leucine-large-blue-gaussian/probability_spectral-blue_{path_cell.stem.partition('binCell_')[2]}.h5")
    list_blues.append(path_blue)

args = pd.DataFrame({
    "path_orga":  list_blues,
    "path_cell":  list_cells,
    "folder_out": "data/rebuttal_error/bycell_blueagain/"
})
batch_apply(calculate_error_blues,args)

# %% READ CALCULATED DATA
# CAUTION: notice the folder name.
list_thresholds = []
for path_csv in Path("data/rebuttal_error/bycell_nonzero").glob("*.csv"):
    list_thresholds.append(pd.read_csv(str(path_csv)))
df_thresholds = pd.concat(list_thresholds)

# %% SHOWED: normal std is not a good idea
scanned_vector = df_thresholds[[f"scanned-{i}" for i in range(100)]].to_numpy()
histogram = np.zeros_like(scanned_vector)
histogram[:,-1] = scanned_vector[:,-1]
histogram[:,:-1] = scanned_vector[:,:-1] - scanned_vector[:,1:]

probabilities = np.linspace(0,1,100,endpoint=False) + 0.01
mu = np.dot(histogram,probabilities)
mean = np.mean(scanned_vector,axis=1)
# >>> np.all(np.all(np.round(mu,2)==np.round(mean,2)))
# True
sdv = np.std(scanned_vector,axis=1)
sigma2 = np.dot((scanned_vector - mean.reshape((-1,1)))**2,probabilities)
sigma  = np.sqrt(sigma2)
# `sigma` should be wrong:
# not make sense to give much higher weight for higher volumes.
z_score = sdv/mean
plt.hist(z_score,bins=50)

# %% Volume Change Percentage vs. Threshold Change
errors_value = scanned_vector - scanned_vector[:,50].reshape((-1,1))
errors_prcnt = errors_value/scanned_vector[:,50].reshape((-1,1))
masked_err_prcnt = np.ma.masked_invalid(errors_prcnt)

for i in range(100):
    df_thresholds[f"pct-{i}"] = masked_err_prcnt[:,i]
by_organelles = df_thresholds.groupby("organelle").mean()

orga_err_pct = {}
for orga in by_organelles.index:
    orga_err_pct[orga] = by_organelles.loc[orga,[f"pct-{i}" for i in range(100)]].to_numpy()
    plt.figure()
    plt.plot(
        0.51 - probabilities[49::-1],
        orga_err_pct[orga][49::-1]
    )
    plt.plot(
        probabilities[51:] - 0.51,
        orga_err_pct[orga][51:]
    )
    plt.title(f"{orga} Volume Change Percentage vs. Threshold Change")
    plt.show()

# %% 
# Overall plot, not very useful because we need to group by organelles
err_avg_prcnt = np.mean(masked_err_prcnt,axis=0)
plt.plot(
    0.51 - probabilities[49::-1],
    err_avg_prcnt[49::-1]
)
plt.plot(
    probabilities[51:] - 0.51,
    err_avg_prcnt[51:]
)
plt.xlim(0,0.4)
plt.ylim(-1,2)

# %% Some metrics I tried, not communicated with advisor.
scanned = scanned_vector[100]

tp = np.zeros(100)
tp[  :50] = scanned[50]
tp[50:  ] = scanned[50:]

fp = np.zeros(100)
fp[  :50] = scanned[:50] - scanned[50]
fp[50:  ] = 0

fn = np.zeros(100)
fn[  :50] = 0
fn[50:  ] = scanned[50] - scanned[50:]

tn = np.zeros(100)
tn[  :50] = scanned[0] - scanned[:50]
tn[50:  ] = scanned[0] - scanned[50]

rate_tp = tp/(tp+fn)
rate_fp = fp/(tn+fp)
specificity = tn/(tn+fp)
f1score = 2*tp/(2*tp + fp + fn)

def robustness_metric(scanned):
    """
    | Abbrev |    Meaning     |     threshold < 0.5      |      threshold > 0.5      |
    |--------|----------------|--------------------------|---------------------------|
    |   TP   |  true positive |       scanned[50]        |        scanned[i]         |
    |   FP   | false positive | scanned[i] - scanned[50] |             0             |
    |   FN   | false negative |            0             | scanned[50] - scanned[ i] |
    |   TN   |  true negative | scanned[0] - scanned[ i] | scanned[ 0] - scanned[50] |
    """
    assert len(scanned)==100, "This function is specific to a single problem, do not over generalize it."
    tp = np.zeros(100)
    tp[  :50] = scanned[50]
    tp[50:  ] = scanned[50:]
    
    fp = np.zeros(100)
    fp[  :50] = scanned[:50] - scanned[50]
    fp[50:  ] = 0

    fn = np.zeros(100)
    fn[  :50] = 0
    fn[50:  ] = scanned[50] - scanned[50:]

    tn = np.zeros(100)
    tn[  :50] = scanned[0] - scanned[:50]
    tn[50:  ] = scanned[0] - scanned[50]

    rate_tp = tp/(tp+fn)
    rate_fp = fp/(tn+fp)

    plt.plot(rate_fp,rate_tp)
    return tp,fp,fn,tn,rate_tp,rate_fp

# %%