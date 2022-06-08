# Organelle Measurements

## How-to

1. Download the project
  ```bash
  git clone git@github.com:ShixingWang/organelle-measure.git
  cd organelle-measure
  ```
2. Install the package required by the scripts
  ```bash
  mkvirtualenv organelle  # if you want to run the project in an isolated environment 
  pip install -e .
  ```
3. Start your IDE _in the proper directory_

  You can use any IDE you like although I use vscode, but it needs to be started in `organelle-measure/` folder for relative paths to work
  ```bash
  code . & 
  ```

## File Structure

- `data/`: output of the pipeline, ignored by git and synced with other tools
  - `ilastik/`: ilastik projects used to segment the images
  - `results/`: csv files containing the cell and organelle properties
  - `figures/`
  - `spectra/`
  - ...
- `images/`: read-only microscopy images, ignored by git and synced with other tools
  - `experiemnt1/`
- `scripts/`: scripts to batch process the images. Often uses `batch_apply()` function in `organelle_measure/`
  - `postprocess_vacuole_allround1data.py`
  - ...
- `notebooks/`: playground to try new methods
  - `notebook1.py`
  - `notebook2.py`
  - ... 
- `test/`: files need by 
- `organelle_measure/`
  - reusable scripts used by notebooks and scripts.
