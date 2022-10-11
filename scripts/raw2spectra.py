# This script prints out the spectrum of the raw images,
# in order to validate our unmixing.

# This file assumes running on the root dir of this project!

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xmltodict
from pathlib import Path
from copy import deepcopy
from skimage import io,util
from organelle_measure.tools import load_nd2_plane,get_nd2_size

def read_spectra_xml(filepath):
    with open(filepath,'rb') as file_xml:
        data_dict = xmltodict.parse(file_xml.read())
    # get the effective key under 'variant'
    count_variant = 0
    for key_variant in data_dict['variant'].keys():
        count_variant += 1
    assert count_variant==2, "Wrong Length of Variant's Children!"

    # get the keys of each individual spectrum
    keys_unmix = []
    count_unmix = 0
    for key_unmix in data_dict["variant"][key_variant].keys():
        if key_unmix[0] != '@':
            count_unmix += 1
            keys_unmix.append(key_unmix)

    # get the data points
    num_points = int(data_dict["variant"][key_variant][key_unmix]["Spectrum"]["uiCount"]["@value"])
    properties = ['eType', 'uiWavelength', 'dWavelength','dTValue']
    node = data_dict["variant"][key_variant][key_unmix]["Spectrum"]["pPoint"][f"Point0"]
    dict_type = {prop:int if "int" in node[prop]["@runtype"] else float for prop in properties}
    list_spectra = []
    for key_unmix in keys_unmix:
        list_spectra.append({prop:np.zeros((num_points,),dtype=dict_type[prop]) for prop in properties})
        for point in range(num_points):
            node = data_dict["variant"][key_variant][key_unmix]["Spectrum"]["pPoint"][f"Point{point}"]
            for key_spectra in list_spectra[-1].keys():
                type_point = int if "int" in node[key_spectra]["@runtype"] else float # how can we not avoid this
                list_spectra[-1][key_spectra][point] = type_point(node[key_spectra]["@value"])
    for i,spectrum in enumerate(list_spectra):
        spectrum["unmix"] = i
    df = pd.concat((pd.DataFrame(spectrum) for spectrum in list_spectra),ignore_index=True)
    return df

def get_spectra_img(file1,file2,file_raw):
    img_label1 = io.imread(str(file1))
    img_label2 = io.imread(str(file2))
    img_raw    = load_nd2_plane(str(file_raw),frame="czyx",axes="t",idx=0)
    # shape = get_nd2_size(str(file_raw))

    label1 = (img_label1>1)
    label2 = (img_label2>1)
    # label0 = np.invert(label1)*np.invert(label2)

    # spectra0 = np.array(img_raw[:,label0]).transpose()
    spectra1 = np.array(img_raw[:,label1]).transpose()
    spectra2 = np.array(img_raw[:,label2]).transpose()

    # max0 = np.max(np.array(spectra0),axis=1)[:,np.newaxis]
    max1 = np.max(np.array(spectra1),axis=1)[:,np.newaxis]
    max2 = np.max(np.array(spectra2),axis=1)[:,np.newaxis]

    # normed0 = np.zeros_like(max0)
    normed1 = np.zeros_like(max1)
    normed2 = np.zeros_like(max2)
    # np.true_divide(1,max0,normed0,where=(max0>0))
    np.true_divide(1,max1,normed1,where=(max1>0))
    np.true_divide(1,max2,normed2,where=(max2>0))
    # normed0 = normed0 * spectra0
    normed1 = normed1 * spectra1
    normed2 = normed2 * spectra2

    # average0 = np.mean(normed0,axis=0)
    average1 = np.mean(normed1,axis=0)
    average2 = np.mean(normed2,axis=0)

    return average1,average2

folder_i = Path("images/raw")
folder_l = Path("images/preprocessed")
folder_o = Path("data/spectra")

subfolders = [
    "EYrainbow_glucose",
    "EYrainbow_rapamycin_1stTry",
    "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st",
    "EYrainbow_leucine_large",
    "EYrainbow_leucine",
    "EYrainbowWhi5Up_betaEstrodiol",
    "EYrainbow_glucose_largerBF"
]

list_meta = []
for subfolder in subfolders:
    for color in ['blue','red']:
        for path_file in (folder_i/subfolder).glob(f"spectral-{color}*.nd2"):
            dict_meta = get_nd2_size(str(path_file))
            dict_meta["folder"] = subfolder
            dict_meta["file"] = path_file.stem
            list_meta.append(dict_meta)
df_meta = pd.concat([pd.DataFrame(meta,index=[m]) for m,meta in enumerate(list_meta)])
df_meta.to_csv("notebooks/meta.csv",index=False)
# The output has been modified to exclude those not suitable for spectrum plot.

# Deal with blue channels of leucine large experiment.
for fileblue in (folder_l/"leucine-large-blue-gaussian").glob("binary-spectral-blue*.tiff"):
    img_old = io.imread(str(fileblue))
    # print(type(img_old),img_old.shape)
    
    img_pex = np.zeros_like(img_old,dtype=int)
    img_pex[img_old==3] = 255
    name_pex = folder_l/"EYrainbow_leucine_large"/f"{fileblue.stem.replace('spectral-blue','peroxisome')}.tiff"
    io.imsave(str(name_pex),util.img_as_ubyte(img_pex))

    img_vph = np.zeros_like(img_old,dtype=int)
    img_vph[img_old==2] = 255
    name_vph = folder_l/"EYrainbow_leucine_large"/f"{fileblue.stem.replace('spectral-blue','vacuole')}.tiff"
    io.imsave(str(name_vph),util.img_as_ubyte(img_vph))

    print(f"Finished: {fileblue.stem}")
# End

dict_files = {
    "blue": folder_o/"EYrainbow_blue_rpmc-0_field-5.xml",
    "red":  folder_o/"EYrainbow_red_glu-200_field-2_try1_largerBF.xml"
}
dict_organelles = {
    "blue":["peroxisome","vacuole"],
    "red": ["mitochondria","LD"]
}

list_benchmark = []
for color in ["blue","red"]:
    df_read = read_spectra_xml(str(dict_files[color]))
    for i in range(2):
        list_benchmark.append(
            pd.DataFrame({
                "wavelength": (df_read.loc[(df_read['eType'].eq(2) & df_read['unmix'].eq(0)),'dWavelength'].to_numpy()
                              +df_read.loc[(df_read['eType'].eq(3) & df_read['unmix'].eq(0)),'dWavelength'].to_numpy()
                              )/2,
                "intensity": df_read.loc[(df_read['unmix'].eq(i) & df_read['eType'].eq(2)),"dTValue"],
                "organelle": dict_organelles[color][i]
            })
        )
df_benchmark = pd.concat(list_benchmark)
df_benchmark_max = df_benchmark.groupby("organelle").max()
df_benchmark.set_index(["organelle",'wavelength'],inplace=True)
df_benchmark["norm"] = df_benchmark.loc[:,"intensity"]/df_benchmark_max.loc[:,'intensity']
df_benchmark.reset_index(inplace=True)

df_meta = pd.read_csv("notebooks/meta.csv")
list_df = []
for folder,stem in zip(df_meta['folder'],df_meta['file']):
    color = 'blue' if 'blue' in stem else 'red'
    
    file_raw = folder_i/folder/f'{stem}.nd2'
    file_label1 = folder_l/folder/f"binary-{dict_organelles[color][0]}_{file_raw.stem.partition('_')[2]}.tiff"
    file_label2 = folder_l/folder/f"binary-{dict_organelles[color][1]}_{file_raw.stem.partition('_')[2]}.tiff"
    
    try:
        mean1,mean2 = get_spectra_img(
                                  str(file_label1),
                                  str(file_label2),
                                  str(file_raw)
                                 )
    except FileNotFoundError:
        print(f"File Not Found: {folder}/{stem}")
        continue
    list_df.append(
        pd.DataFrame({
            "organelle":     (orga:=dict_organelles[color][0]),
            "wavelength":    df_benchmark.loc[df_benchmark['organelle'].eq(orga),'wavelength'].to_list()[:len(mean1)],
            "intensity":     mean1,
            "experiment":    folder,
            "field_of_view": stem
        })        
    )
    list_df.append(
        pd.DataFrame({
            "organelle":     (orga:=dict_organelles[color][1]),
            "wavelength":    df_benchmark.loc[df_benchmark['organelle'].eq(orga),'wavelength'].to_list()[:len(mean1)],
            "intensity":     mean2,
            "experiment":    folder,
            "field_of_view": stem
        })        
    )
df_img = pd.concat(list_df)
df_img.to_csv("data/spectra/spectra_images.csv",index=False)


# Plot spectra:

df_img = pd.read_csv("data/spectra/spectra_images.csv")
df_exp = df_img.groupby(['experiment','organelle','wavelength']).mean()
df_exp.reset_index(inplace=True)

df_max = df_exp.groupby(['experiment','organelle']).max()
df_max.reset_index(inplace=True)

df_exp.set_index(['experiment','organelle'],inplace=True)
df_max.set_index(['experiment','organelle'],inplace=True)

df_exp['norm'] = df_exp.loc[:,'intensity']/df_max.loc[:,'intensity']
df_exp.reset_index(inplace=True)

# color = 'blue'
for color in ['blue','red']:
    fig = go.Figure()
    for experiment in df_exp['experiment'].unique():
        if experiment == "EYrainbow_leucine_large":
            continue
        fig.add_trace(
            go.Scatter(
                x = df_exp.loc[(df_exp["experiment"].eq(experiment) & df_exp["organelle"].eq(dict_organelles[color][0])),'wavelength'],
                y = df_exp.loc[(df_exp["experiment"].eq(experiment) & df_exp["organelle"].eq(dict_organelles[color][0])),'norm'],
                name = f"{dict_organelles[color][0]}: {experiment}",
                mode="lines+markers",
                line=dict(dash="solid",shape="spline",color='grey')
            )
        )
        fig.add_trace(
            go.Scatter(
                x = df_exp.loc[(df_exp["experiment"].eq(experiment) & df_exp["organelle"].eq(dict_organelles[color][1])),'wavelength'],
                y = df_exp.loc[(df_exp["experiment"].eq(experiment) & df_exp["organelle"].eq(dict_organelles[color][1])),'norm'],
                name = f"{dict_organelles[color][1]}: {experiment}",
                mode="lines+markers",
                line=dict(dash="dash",shape="spline",color='grey')
            )
        )
    fig.add_trace(
        go.Scatter(
            x = df_benchmark.loc[(df_benchmark["organelle"].eq(dict_organelles[color][0])),'wavelength'],
            y = df_benchmark.loc[(df_benchmark["organelle"].eq(dict_organelles[color][0])),'norm'],
            name = dict_organelles[color][0],
            mode="lines+markers",
            line=dict(dash="solid",shape="spline",color='blue')
        )
    )
    fig.add_trace(
        go.Scatter(
            x = df_benchmark.loc[(df_benchmark["organelle"].eq(dict_organelles[color][1])),'wavelength'],
            y = df_benchmark.loc[(df_benchmark["organelle"].eq(dict_organelles[color][1])),'norm'],
            name = dict_organelles[color][1],
            mode="lines+markers",
            line=dict(dash="solid",shape="spline",color='red')
        )
    )
    fig.update_layout(template="simple_white")
    fig.write_html(f"{folder_o}/white_unmix_comparison_{color}.html")