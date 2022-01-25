# The Overview of Organelle Science: 

- Image Acquisition
- Segmentation
- Measurements
- Analysis

The Segmentation part is centered by ilastik. The inputs are acquired images, the outputs are the binary images:
- open files & preprocessing
- ilastik
- postprocessing

Measurement part: the inputs are the binary images, the outputs are the csv files.

Analysis part, the inputs are the csv files, and the outputs are the plots. 

## File Open and Preprocess

Opening a file has been well done with `ND2Reader`.

The preprocessing has only been tested on the 2 blue channel organelles: `gaussian(sigma=1.0)` is good.