# Compare the segmentation result of vph1 with Gaussian filters of different sigma

In order to find the volumeric vacuoles, some time ago I tried the first half and propose the following:
1. using the ilastik to do the segmentation of vph1 signals
2. using imageJ/python to skeletonize the binary images
3. using python binary 3D dilation/closing for
4. using python to flood fill 2D the closed ring structures
5. (optional) assuming there can be only one vacuole accross different z, and fit with a cube.

I'm trying segmenting the step1 with different sigmas for the gaussian filter and compare the effect.

- bkgd_vph1-vph1_gaussian03_1nmpp1TriK_field2.ilp: sigma=0.3 gives too many salt and smoke noises in ilastik. Also this proj does not save 
- bkgd_vph1-vph1_gaussian05_1nmpp1TriK_field2.ilp: sigma=0.5 don't like it very much
- bkgd_vph1-vph1_gaussian08_1nmpp1TriK_field2.ilp: sigma=0.8 like the result very much
- bkgd_vph1-vph1_gaussian1_1nmpp1TriK_field2.ilp: do not see so many rings by eyes, actually none :-(
- bkgd_vph1-vph1_gaussian15_1nmpp1TriK_field2.ilp: sigma=1.5 very bad, the hole in the center is almost invisible