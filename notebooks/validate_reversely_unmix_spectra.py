# This script validates that the segmented sec7 signals are really lipid droplet
# - (shall we) preprocess the unmixed red images
#     1. with gaussian blurring
#     2. without gaussian blurring
# - use ilastik to segment the image
# - read the spectra of the segmented lipid droplets from spectral images
# - measure the performance:
#     1. similarity with the theoretical spectra: mutual information
#     2. uniformity: the variance of the mutual information
#     3. difference between the background and foreground
