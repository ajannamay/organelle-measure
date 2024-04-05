# Monte Carlo simulation of the organelle volumes in a cell.
# The issue of rebuttal_error_bycell is that it scans through unique probabilities in the images, thus the possible total volume have a very limited number of possible values
# This script cope with this problem by generating a random array and compare it with the probability image given by `ilastik`.
# By repeating this for a ot of times, we have a Monte-Carlo estimate of the distribution of the volume, which hopefully is rather continuous.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,util,measure
from scipy import stats
import h5py

N_SAMPLE = 10000

organelles = [
    "peroxisome",
    "vacuole"
    "ER",
    "golgi",
    "mitochondria",
    "LD",
]

# path_cell = "images/cell/paperRebuttal/binCell_EYrainbow_glu-100_field-4.tif"
# path_organelle = "images/preprocessed/paperRebuttal/probability_mitochondria_EYrainbow_glu-100_field-4.h5"

# %%
path_cell = "images/cell/EYrainbow_glucose_largerBF/binCell_EYrainbow_glu-100_field-4.tif"

for organelle in organelles:

    path_organelle = f"images/preprocessed/EYrainbow_glucose_largerBF/probability_{organelle}_EYrainbow_glu-100_field-4.h5"


    img_cell = io.imread(path_cell)

    file_orga = h5py.File(path_organelle,"r")
    img_orga = file_orga["exported_data"][1] # type:ignore

    # TRY FOR A SINGLE CELL (PREVIOUS JYPYTER CELL)
    dfs = []
    for cell in measure.regionprops(img_cell):
        min_row, min_col, max_row, max_col = cell.bbox
        img_cell_crop = cell.image
        img_orga_crop = img_orga[:,min_row:max_row,min_col:max_col] # type:ignore
        for z in range(img_orga_crop.shape[0]):
            img_orga_crop[z] = img_orga_crop[z] * img_cell_crop # type:ignore
        sampled_sizes = np.zeros(N_SAMPLE)
        for i in range(N_SAMPLE):
            randoms = np.random.random_sample(size=img_orga_crop.shape) # type:ignore
            sampled_sizes[i] = np.count_nonzero(np.round(img_orga_crop,2)>randoms) # type:ignore
        goodness = stats.goodness_of_fit(
            stats.norm,sampled_sizes
        )
        measured = {
            "loc"      : [goodness.fit_result.params.loc],
            "scale"    : [goodness.fit_result.params.scale],
            "statistic": [goodness.statistic],
            "p_value"  : [goodness.pvalue],
            "segment"  : [np.count_nonzero(img_orga_crop>0.5)], # type:ignore
            "z_score"  : [(np.count_nonzero(img_orga_crop>0.5) - goodness.fit_result.params.loc)/goodness.fit_result.params.scale] # type:ignore
            
        }
        dfs.append(pd.DataFrame(measured))
        # plt.hist(sampled_sizes,bins=50,density=True)
        # xticks = np.linspace(sampled_sizes.min(),sampled_sizes.max(),100)
        # plt.plot(
        #     xticks,
        #     stats.norm.pdf(xticks,
        #         loc  =goodness.fit_result.params.loc,
        #         scale=goodness.fit_result.params.scale
        #     )
        # )
        print(f"... finished process Cell #{cell.label}")
    df = pd.concat(dfs,ignore_index=True)
    df.to_csv(
        f"data/rebuttal_error/MonteCarlo_{organelle}_EYrainbow_glu-100_field-4.csv",
        index=False
    )
    print(f"Finished processing {organelle}")


# %%
df_read = {}
for organelle in organelles:
    df_read[organelle] = pd.read_csv(f"data/rebuttal_error/MonteCarlo_{organelle}_EYrainbow_glu-100_field-4.csv")
    plt.figure()
    plt.scatter(
        df_read[organelle]["segment"],
        df_read[organelle]["loc"]
    )
    x_min = df_read[organelle]["segment"].min()
    x_max = df_read[organelle]["segment"].max()
    plt.plot(
        np.linspace(x_min*0.9,x_max*1.1),
        np.linspace(x_min*0.9,x_max*1.1)
    )
    plt.axis('equal')
    plt.title(organelle)
    plt.xlabel("Segmented Size")
    plt.ylabel("Monte Carlo Estimate")
    plt.savefig(f"data/rebuttal_error/MonteCarlo_segment_{organelle}.png")


# %%
