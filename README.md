# Organelle Measurements

## Preparation

1. Required softwares:
  - YeaZ-GUI
  - ilastik
  - Python
    - nd2reader
    - numpy
    - pandas
    - scipy
    - scikit-image
    - scikit-learn
    - matplotlib
    - seaborn
    - plotly (optional)
2. Download the project
  ```bash
  git clone git@github.com:ShixingWang/organelle-measure.git
  cd organelle-measure
  ```
3. Install the package required by the scripts
  ```bash
  mkvirtualenv organelle  # if you want to run the project in an isolated environment 
  pip install -e .
  ```
4. Start your IDE _in the proper directory_

  You can use any IDE you like although I use vscode, but it needs to be started in `organelle-measure/` folder for relative paths to work.
  ```bash
  cd path/to/organelle-measure
  code . & 
  ```

## File Structure

- `organelle_measure/`
  - reusable scripts used by notebooks and scripts.
- `notebooks/`: playground to try new methods
  - `notebook1.py`
  - `notebook2.py`
  - ... 
- `test/`: files need by `notebooks/`
- `scripts/`: scripts to batch process the images. Often uses `batch_apply()` function in `organelle_measure/`
  - `postprocess_vacuole_allround1data.py`
  - ...
- `images/`
  - `raw`: read-only microscopy images, ignored by git and synced by another software
    - `EYrainbow_glucose_largerBF`
    - `EYrainbowWhi5Up_betaEstrodiol`
    - `EYrainbow_leucine_large`
    - `EYrainbow_1nmpp1_1st`
    - `EYrainbow_rapamycin_1stTry`
    - `EYrainbow_rapamycin_CheckBistability`
      - `f"camera-{before/after}_EYrainbow_{experiment}-{condition}_field-{f}.nd2"`
      - ...
      - `f"spectral-{blue/green/yellow/red}_EYrainbow_{experiment}-{condition}_field-{f}.nd2"`
      - ...
  - `cell/`
    - `EYrainbow_glucose_largerBF`
    - `EYrainbowWhi5Up_betaEstrodiol`
    - `EYrainbow_leucine_large`
    - `EYrainbow_1nmpp1_1st`
    - `EYrainbow_rapamycin_1stTry`
    - `EYrainbow_rapamycin_CheckBistability`
      - `f"binCell_EYrainbow_{experiment}-{condition}_field-{f}.tif"`
      - ...
  - `preprocessed/`
    - `EYrainbow_glucose_largerBF`
    - `EYrainbowWhi5Up_betaEstrodiol`
    - `EYrainbow_leucine_large`
    - `EYrainbow_1nmpp1_1st`
    - `EYrainbow_rapamycin_1stTry`
    - `EYrainbow_rapamycin_CheckBistability`
      - `f"{organelle}_EYrainbow_{experiment}-{condition}_field-{f}.tif"`
      - ...
      - `f"probability_{organelle}_EYrainbow_{experiment}-{condition}_field-{f}.tif"`
      - ...
    - `leucine-large-blue-gaussian`
      - `f"probability_spectral-blue_EYrainbow_leu-0_hour-3,field-{f}.h5"`
      - ...
  - `labelled/`
    - `EYrainbow_glucose_largerBF`
    - `EYrainbowWhi5Up_betaEstrodiol`
    - `EYrainbow_leucine_large`
    - `EYrainbow_1nmpp1_1st`
    - `EYrainbow_rapamycin_1stTry`
    - `EYrainbow_rapamycin_CheckBistability`
      - `f"label-{organelle}_EYrainbow_{experiment}-{condition}_field-{f}.tiff"`
- `data/`: output of the pipeline, ignored by git and synced by another software
  - `ilastik/`: ilastik projects used to segment the images
  - `results/`
    - `EYrainbow_glucose_largerBF`
    - `EYrainbowWhi5Up_betaEstrodiol`
    - `EYrainbow_leucine_large`
    - `EYrainbow_1nmpp1_1st`
    - `EYrainbow_rapamycin_1stTry`
    - `EYrainbow_rapamycin_CheckBistability`
      - `f"cell_EYrainbow_{experiment}-{condition}_field-{f}.csv"`
      - `f"{organelle}_EYrainbow_{experiment}-{condition}_field-{f}.csv"`
  - `figures/`
  - `spectra/`
  - ...

## Pipeline

1. 


