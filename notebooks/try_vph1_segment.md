## Compare the segmentation result of vph1 with Gaussian filters of different sigma

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

## Decided pipeline with `ilastik`

Here is an update on better recognizing the vacuole project.

I have uploaded the sample image onto box at `OrganelleAnalysis/TryThresholdVacuole`

Difficulties are:
- peroxisomes are showing in the unmixed vacuole channel as dot false positives. This is very common.
- in some planes vacuole signals do not form a closed ring.
- the image quality

The pipeline is as follows:
- start with the unmixed-**.nd2 image
- apply a difference of gaussians filter: vph1_diffgaussian-**.tif
- use ilastika to binarize the image: bin-vph1-**.tif
- skeletonize the image: skeleton-vph1-**.tif
- flood fill the image so that closed ring will be recognized as vacuoles: skeleton-fill-vph1-**.tif

This way gives false negatives because many nearly closed rings are not filled. But if we assume there can be only one vacuole at one x-y coordinate, then we have a better and confident counting number of the vacuoles.

As for the volume I'm thinking about 2 ways:
- estimate the volume by finding the largest ring and assume it is a sphere
- start from where there are vacuoles, scan across all z and nearby x and y to find those nearly closed rings, try convex hull to fill the hole. This might not work if there are 2 connecting vacuole boundaries.