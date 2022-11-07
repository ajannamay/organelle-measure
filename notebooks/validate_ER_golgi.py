import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from skimage import util,io,morphology,filters
from organelle_measure import tools

import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)

for nd2 in Path("data/Validate6colorWT/nd2/").glob("*.nd2"):
    print(nd2,nd2.stem)
    image = bioformats.load_image(str(nd2), c=None, z=0, t=0, series=None, index=None, rescale=False, wants_max_intensity=False, channel_names=None)
    image = np.transpose(image,[2,0,1])
    # image = tools.load_nd2_plane(str(nd2),frame="cyx",axes="t",idx=0).astype(int)
    io.imsave(f"data/Validate6colorWT/tif/{nd2.stem.replace(' ','_')}.tif",util.img_as_uint(image))

javabridge.kill_vm()



img_er1 = io.imread("data/Validate6colorWT/tif/488_ER.tif")
img_er2 = io.imread("data/Validate6colorWT/tif/488_ER_2.tif")
img_gg1 = io.imread("data/Validate6colorWT/tif/514_golgi_.tif")
img_gg2 = io.imread("data/Validate6colorWT/tif/514_golgi_2.tif")

gauss_gg1 = np.zeros_like(img_gg1)
gauss_gg2 = np.zeros_like(img_gg2)
for c in range(gauss_gg1.shape[0]):
    gauss_gg1[c] = filters.gaussian(img_gg1[c],sigma=0.5)
for c in range(gauss_gg2.shape[0]):
    gauss_gg2[c] = filters.gaussian(img_gg2[c],sigma=0.5)
io.imsave("data/Validate6colorWT/tif/gaussian_514_golgi.tif",gauss_gg1)
io.imsave("data/Validate6colorWT/tif/gaussian_514_golgi_2.tif",gauss_gg2)

mask_er1 = io.imread("data/Validate6colorWT/segment/Simple Segmentation_488_ER.tif")
mask_er2 = io.imread("data/Validate6colorWT/segment/Simple Segmentation_488_ER_2.tif")
mask_gg1 = io.imread("data/Validate6colorWT/segment/Simple Segmentation_SUM_514_golgi.tif")
mask_gg2 = io.imread("data/Validate6colorWT/segment/Simple Segmentation_514_golgi_2.tif")

mask_er1 = (mask_er1==2)
mask_gg1 = (mask_gg1==2)
# mask_er2 = (mask_er2==2)
# mask_gg2 = (mask_gg2==2)

mask_er1 = morphology.binary_erosion(mask_er1)
# mask_gg1 = morphology.binary_erosion(mask_gg1)
# mask_er2 = morphology.binary_erosion(mask_er2)
# mask_gg2 = morphology.binary_erosion(mask_gg2)

spectra_er1_er = np.transpose(img_er1[:,mask_er1])
spectra_er1_gg = np.transpose(img_er1[:,mask_gg1])
# spectra_er2_er = np.transpose(img_er2[:,mask_er2])
# spectra_er2_gg = np.transpose(img_er2[:,mask_gg2])

spectra_gg1_er = np.transpose(img_gg1[:,mask_er1])
spectra_gg1_gg = np.transpose(img_gg1[:,mask_gg1])
# spectra_gg2_er = np.transpose(img_gg2[:,mask_er2])
# spectra_gg2_gg = np.transpose(img_gg2[:,mask_gg2])


spectrum_er1_er = np.mean(spectra_er1_er,axis=0)
spectrum_er1_gg = np.mean(spectra_er1_gg,axis=0)

spectrum_gg1_er = np.mean(spectra_gg1_er,axis=0)
spectrum_gg1_gg = np.mean(spectra_gg1_gg,axis=0)

# spectrum_er2_er = np.mean(spectra_er2_er,axis=0)
# spectrum_er2_gg = np.mean(spectra_er2_gg,axis=0)

# spectrum_gg2_er = np.mean(spectra_gg2_er,axis=0)
# spectrum_gg2_gg = np.mean(spectra_gg2_gg,axis=0)

wavelengths_g_left  = np.array([495.7,501.7,507.7,513.7,519.7,525.7,531.8,537.8,543.8,549.8,555.8,561.8])
wavelengths_g_right = np.array([500.5,506.5,512.5,518.5,524.5,530.5,536.5,542.6,548.6,554.6,560.6,566.6])
wavelengths_g = (wavelengths_g_left + wavelengths_g_right)/2

# wavelengths_y_left  = np.array([521.1,527.1,533.1,539.1,545.1,551.1,557.1])
# wavelengths_y_right = np.array([525.9,531.9,537.9,543.9,549.9,555.9,561.9])
wavelengths_y_left  = np.array([520.9,526.9,532.9,538.9,544.9,550.9,557.0,563.0,569.0,575.0,585.8,587.0])
wavelengths_y_right = np.array([525.7,531.7,537.7,543.7,549.7,555.8,561.8,567.8,573.8,579.8,581.0,591.8])
wavelengths_y = (wavelengths_y_left + wavelengths_y_right)/2

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = wavelengths_g[:-3],
        y = spectrum_er1_er[:-3]/np.max(spectrum_er1_er[:-3]),
        name = "ER",
        mode="lines+markers",
        line=dict(dash="solid",shape="spline",color='blue')
    )
)
fig.add_trace(
    go.Scatter(
        x = wavelengths_g[:-3],
        y = spectrum_er1_gg[:-3]/np.max(spectrum_er1_er[:-3]),
        name = "Golgi",
        mode="lines+markers",
        line=dict(dash="solid",shape="spline",color='red')
    )
)
fig.update_layout(template="simple_white")
fig.write_html("data/spectra/spectra_ER.html")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = wavelengths_y[:-5],
        y = spectrum_gg1_er[:-5]/np.max(spectrum_gg1_gg[:-5]),
        name = "ER",
        mode="lines+markers",
        line=dict(dash="solid",shape="spline",color='blue')
    )
)
fig.add_trace(
    go.Scatter(
        x = wavelengths_y[:-5],
        y = spectrum_gg1_gg[:-5]/np.max(spectrum_gg1_gg[:-5]),
        name = "Golgi",
        mode="lines+markers",
        line=dict(dash="solid",shape="spline",color='red')
    )
)
fig.update_layout(template="simple_white")
fig.write_html("data/spectra/spectra_Golgi.html")

# # Use a different folder

# img_er = io.imread("data/ValidateErGolgi/tif/LeuExp100GFP_01-1.tif")
# img_gg = io.imread("data/ValidateErGolgi/tif/LeuExp100YFP_01-1.tif")

# mask_er = io.imread("data/ValidateErGolgi/tif/Unmixing_green_ER-1.tif").astype(bool)
# mask_gg = io.imread("data/ValidateErGolgi/tif/Unmixing_yellow_Golgi-1.tif").astype(bool)

# spectra_er_er = np.transpose(img_er[:,mask_er])
# spectra_er_gg = np.transpose(img_er[:,np.logical_and(mask_gg,np.logical_not(mask_er))])

# spectra_gg_er = np.transpose(img_gg[:,np.logical_and(mask_er,np.logical_not(mask_gg))])
# spectra_gg_gg = np.transpose(img_gg[:,mask_gg])

# spectrum_er_er = np.mean(spectra_er_er,axis=0)
# spectrum_er_gg = np.mean(spectra_er_gg,axis=0)

# spectrum_gg_er = np.mean(spectra_gg_er,axis=0)
# spectrum_gg_gg = np.mean(spectra_gg_gg,axis=0)
