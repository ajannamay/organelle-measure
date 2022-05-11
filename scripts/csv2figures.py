import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

px_x,px_y,px_z = 0.41,0.41,0.20

organelles = [
    "peroxisome",
    "vacuole",
    "ER",
    "golgi",
    "mitochondria",
    "LD"
]

subfolders = [
    "EYrainbow_glucose",
    "EYrainbow_glucose_largerBF",
    "EYrainbow_rapamycin_1stTry",
    "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st",
    "EYrainbow_leucine_large",
    "EYrainbow_leucine",
    "EYrainbowWhi5Up_betaEstrodiol"
]

folder_i = Path("./data/results")
folder_o = Path("./data/figures/pairwise")
folder_pca = Path("./data/figures/pca_transparent_bkgd")


# READ FILES

dfs_cell = []
dfs_orga = []
for folder in subfolders:
    df_folder_cell = pd.concat((pd.read_csv(fcell) for fcell in (Path(folder_i)/folder).glob("cell*.csv")))
    df_folder_orga = pd.concat((pd.read_csv(fcell) for fcell in (Path(folder_i)/folder).glob("[!c]*.csv")))

    df_folder_cell["folder"] = folder
    df_folder_orga["folder"] = folder
    
    dfs_cell.append(df_folder_cell)
    dfs_orga.append(df_folder_orga)


df_cell_all = pd.concat(dfs_cell)
df_orga_all = pd.concat(dfs_orga)

df_cell_all["condition"] = df_cell_all["condition"].apply(lambda x:float(str(x).replace('-',".")))
df_orga_all["condition"] = df_orga_all["condition"].apply(lambda x:float(str(x).replace('-',".")))

type_cell = {
    "folder":     "string",
    "experiment": "string",
    "condition":  "float",
    "hour":       "int8",
    "field":      "int8",
    "idx-cell":   "int16",
    "area":       "int16",
    "bbox-0":     "int16",
    "bbox-1":     "int16",
    "bbox-2":     "int16",
    "bbox-3":     "int16",
}
type_orga = {
    "folder":       "string",
    "experiment":   "string",
    "condition":    "float",
    "hour":         "int8",
    "field":        "int8",
    "organelle":    "string",
    "idx-cell":     "int16",
    "idx-orga":     "int16",
    "volume-pixel": "int16",
    "volume-bbox":  "int16",
    "bbox-0":       "int16",
    "bbox-1":       "int16",
    "bbox-2":       "int16",
    "bbox-3":       "int16",
    "bbox-4":       "int16",
    "bbox-5":       "int16"
}

df_cell_all = df_cell_all.astype(type_cell)
df_orga_all = df_orga_all.astype(type_orga)


# GROUP BY CELL
df_cell_all.loc[:,"effective-volume"] = (px_x*px_y)*np.sqrt(px_x*px_y)*(2.)*df_cell_all.loc[:,"area"]*np.sqrt(df_cell_all.loc[:,"area"])/np.sqrt(np.pi) 
pivot_cell_bycell = df_cell_all.set_index(["folder","condition","field","idx-cell"])

# # (DEPRECATED) data (in unit of pixels)
# pivot_orga_bycell_mean = df_orga_all.loc[:,["folder","condition","field","organelle","idx-cell","volume-pixel"]].groupby(["folder","condition","field","idx-cell","organelle"]).mean()["volume-pixel"]
# pivot_orga_bycell_nums = df_orga_all.loc[:,["folder","condition","field","organelle","idx-cell","volume-pixel"]].groupby(["folder","condition","field","idx-cell","organelle"]).count()["volume-pixel"]
# pivot_orga_bycell_totl = df_orga_all.loc[:,["folder","condition","field","organelle","idx-cell","volume-pixel"]].groupby(["folder","condition","field","idx-cell","organelle"]).sum()["volume-pixel"]

# data (in unit of microns)
df_orga_all["volume-micron"] = np.empty_like(df_orga_all.index)
df_orga_all.loc[df_orga_all["organelle"].eq("vacuole"),"volume-micron"] = (px_x*px_y)*np.sqrt(px_x*px_y)*(2.)*df_orga_all.loc[df_orga_all["organelle"].eq("vacuole"),"volume-pixel"]*np.sqrt(df_orga_all.loc[df_orga_all["organelle"].eq("vacuole"),"volume-pixel"])/np.sqrt(np.pi) 
df_orga_all.loc[df_orga_all["organelle"].ne("vacuole"),"volume-micron"] = px_x*px_y*px_z*df_orga_all.loc[df_orga_all["organelle"].ne("vacuole"),"volume-pixel"]

pivot_orga_bycell_mean = df_orga_all.loc[:,["folder","condition","field","organelle","idx-cell","volume-micron"]].groupby(["folder","condition","field","idx-cell","organelle"]).mean()["volume-micron"]
pivot_orga_bycell_nums = df_orga_all.loc[:,["folder","condition","field","organelle","idx-cell","volume-micron"]].groupby(["folder","condition","field","idx-cell","organelle"]).count()["volume-micron"]
pivot_orga_bycell_totl = df_orga_all.loc[:,["folder","condition","field","organelle","idx-cell","volume-micron"]].groupby(["folder","condition","field","idx-cell","organelle"]).sum()["volume-micron"]
# index
index_bycell = pd.MultiIndex.from_tuples(
    [(*index,orga) for index in pivot_cell_bycell.index.to_list() for orga in organelles],
    names=['folder','condition','field','idx-cell','organelle']
)
pivot_bycell = pd.DataFrame(index=index_bycell)
# combine data with index
pivot_bycell.loc[pivot_orga_bycell_mean.index,"mean"] = pivot_orga_bycell_mean
pivot_bycell.loc[pivot_orga_bycell_mean.index,"count"] = pivot_orga_bycell_nums
pivot_bycell.loc[pivot_orga_bycell_mean.index,"total"] = pivot_orga_bycell_totl
# pivot_bycell.fillna(0.,inplace=True)

# include cell data
pivot_bycell.reset_index("organelle",inplace=True) # comment out after 1st run 
pivot_bycell.loc[:,"cell-area"] = pivot_cell_bycell.loc[:,"area"]
pivot_bycell.loc[:,"cell-volume"] = pivot_cell_bycell.loc[:,"effective-volume"]
pivot_bycell.loc[:,"total-fraction"] = pivot_bycell.loc[:,"total"]/pivot_bycell.loc[:,"cell-volume"]
# clean up
df_bycell = pivot_bycell.reset_index()
df_bycell.loc[(df_bycell["organelle"].eq("vacuole") & df_bycell["total"].eq(0.)),"count"] = np.nan
df_bycell.loc[(df_bycell["organelle"].eq("vacuole") & df_bycell["total"].eq(0.)),"mean"] = np.nan
df_bycell.loc[(df_bycell["organelle"].eq("vacuole") & df_bycell["total"].eq(0.)),"total"] = np.nan
pivot_bycell = df_bycell.set_index(index_bycell)


# DATAFRAME FOR CORRELATION COEFFICIENT
for folder in subfolders:
    pv_bycell = df_bycell[df_bycell['folder'].eq(folder)].set_index(['condition','field','idx-cell'])
    df_corrcoef = pd.DataFrame(index=pivot_cell_bycell.loc[folder,:].index)
    df_corrcoef.loc[:,'effective-length'] = np.sqrt(pivot_cell_bycell.loc[folder,'area']/np.pi)
    df_corrcoef.loc[:,'cell-area'] = pivot_cell_bycell.loc[folder,'area']
    df_corrcoef.loc[:,'cell-volume'] = pivot_cell_bycell.loc[folder,'effective-volume']
    properties = []
    for orga in organelles:
        for prop in ['mean','count','total','total-fraction']:
            if (orga=="ER") and (prop in ["count","mean"]):
                continue
            if (orga=="vacuole") and (prop in ["count","mean"]):
                continue
            prop_new = f"{prop}-{orga}"
            properties.append(prop_new)
            df_corrcoef.loc[:,prop_new] = pv_bycell.loc[pv_bycell["organelle"]==orga,prop]
    df_corrcoef.reset_index(inplace=True)

    # # Pairwise relation atlas
    # fig_pair = sns.PairGrid(df_corrcoef,hue="condition",vars=['effective-length','cell-area','cell-volume',*properties],height=3.0)
    # fig_pair.map_diag(sns.histplot)
    # # f_pairig.map_offdiag(sns.scatterplot)
    # fig_pair.map_upper(sns.scatterplot)
    # fig_pair.map_lower(sns.kdeplot)
    # fig_pair.add_legend()
    # fig_pair.savefig(f"{folder_o}/pairplot_{folder}.png")

    # Pairwise relation individual
    sns.set_palette(sns.blend_palette(['red','blue']))
    g = sns.jointplot(
                      data=df_corrcoef, 
                      x="cell-volume", y="total-fraction-mitochondria",
                      hue="condition", kind="kde"
        )
    # g.plot_joint(sns.kdeplot, zorder=0, levels=6)
    g.plot_marginals(sns.rugplot, height=-.15, clip_on=False)

    # # Correlation coefficient
    # np_corrcoef = df_corrcoef.loc[:,['condition','effective-length','cell-area','cell-volume',*properties]].to_numpy()
    # corrcoef = np.corrcoef(np_corrcoef,rowvar=False)
    # fig = px.imshow(
    #         corrcoef,
    #         x=['condition','effective-length','cell-area','cell-volume',*properties],
    #         y=['condition','effective-length','cell-area','cell-volume',*properties],
    #         color_continuous_scale = "RdBu_r",range_color=[-1,1]
    #     )
    # fig.write_html(f"{folder_o}/corrcoef_{folder}.html")

    # for condi in df_corrcoef["condition"].unique():
    #     np_corrcoef = df_corrcoef.loc[df_corrcoef['condition']==condi,['effective-length','cell-area','cell-volume',*properties]].to_numpy()
    #     corrcoef = np.corrcoef(np_corrcoef,rowvar=False)
    #     fig = px.imshow(
    #             corrcoef,
    #             x=['effective-length','cell-area','cell-volume',*properties],
    #             y=['effective-length','cell-area','cell-volume',*properties],
    #             color_continuous_scale="RdBu_r",range_color=[-1,1]
    #         )
    #     fig.write_html(f"{folder_o}/corrcoef_{folder}_{str(condi).replace('.','-')}.html")    

# One layer deeper than correlation coefficient


# EXPERIMENT CONDITION LEVEL

pivot_bycondition = df_bycell.groupby(['folder','organelle','condition']).mean()[['mean','count','total','cell-area','cell-volume','total-fraction']]
pivot_bycondition["cell_count"] = df_bycell[['folder','organelle','condition','mean']].groupby(['folder','organelle','condition']).count()
df_bycondition = pivot_bycondition.reset_index()


# PLOTS

def plot_histo_violin(df,prop_y,prop_x="cell_area",savepath="histo-violin.html"):
    df["plotly"] = df[prop_y].rank(pct=True) # `pct` works but fuck why
    fig = make_subplots(rows=1,cols=1)
    fig.append_trace(
        go.Violin(
            y=df["plotly"],
            # y=np.arange(len(df[prop_y].unique())),
            x=df[prop_x]
        ),
        row=1, col=1
    )
        
    wid = 2./len(df["plotly"].unique())
    
    fig.update_traces(orientation='h', side='positive', width=wid, points="all",pointpos=0.5,jitter=0.4,line_color="deepskyblue",marker_color="tomato",marker_opacity=0.1,row=1, col=1)
    # print(df[prop_y].unique())
    # print(df["plotly"].unique())
    fig.update_layout(
                    title=Path(savepath).stem.replace("_"," "),
                    yaxis_title=prop_y,
                    yaxis_ticktext=df[prop_y].unique(),
                    yaxis_tickvals=df["plotly"].unique(),
                    xaxis_title=f'{prop_x}'
    )
    fig.show()
    fig.write_html(str(savepath))
    return None

def plot_histo_box(df,prop_y,prop_x="cell_area",savepath=""):
    fig = make_subplots(rows=1,cols=1)
    for prop in df[prop_y].unique():
        fig.append_trace(
            go.Box(x=df.loc[df[prop_y]==prop,prop_x],
                   boxmean='sd'),
            row=1, col=1
        )
    
    fig.update_layout(
                    title=Path(savepath).stem.replace("_"," "),
                    yaxis_title=prop_y,
                    yaxis_ticktext=df[prop_y].unique(),
                    yaxis_tickvals=df[prop_y].unique(),
                    xaxis_title=f'{prop_x}'
    )
    fig.show()
    fig.write_html(str(savepath))
    return None

for folder in subfolders:
    for orga in organelles:
        plot_histo_violin(
            df_bycell.loc[df_bycell["organelle"].eq(orga) & df_bycell["folder"].eq(folder)]
            ,"condition","mean",
            f"{folder_o}/cellular_mean_vs_condition_{folder}_{orga}.html"
        )
        plot_histo_violin(
            df_bycell.loc[df_bycell["organelle"].eq(orga) & df_bycell["folder"].eq(folder)],
            "condition","count",
            f"{folder_o}/cellular_count_vs_condition_{folder}_{orga}.html"
        )


def plot_group_hist(df,prop_y,prop_x="cell_area",savepath=""):
    fig = make_subplots(rows=1,cols=1)
    for i,prop in enumerate(df[prop_y].unique()):
        fig.append_trace(
            go.Histogram(
                x=df.loc[df[prop_y].eq(prop),prop_x],
                name=df[prop_y].unique()[i]),
            row=1, col=1
        )
    
    fig.update_layout(
                    barmode='overlay',
                    title=Path(savepath).stem.replace("_"," "),
                    yaxis_title=prop_y,
                    xaxis_title=f'{prop_x}',
                    bargap=0.001,bargroupgap=0.05
    )
    fig.update_traces(opacity=0.75)
    # fig.show()
    fig.write_html(str(savepath))
    return None

for folder in subfolders:
    for orga in organelles:
        plot_group_hist(
            df_bycell.loc[df_bycell["organelle"].eq(orga) & df_bycell["folder"].eq(folder)],
            "condition","mean",
            f"{folder_o}/organelle_mean_vs_condition_{folder}_{orga}.html"
        )


from sklearn.decomposition import PCA

# PCA of volume fraction, and cell size
def make_pca_plots(property,has_volume=False,is_normalized=False):
    name = f"{'cell-volume' if has_volume else 'organellle-only'}_{property}_{'norm-mean-std' if is_normalized else 'raw'}"
    dict_explained_variance_ratio = {}
    for folder in subfolders:
        df_orga_perfolder = df_bycell[df_bycell["folder"].eq(folder)].set_index(["condition","field","idx-cell"])
        idx = df_orga_perfolder.groupby(["condition","field","idx-cell"]).count().index
        df_pca = pd.DataFrame(index=idx,columns=organelles)
        
        columns = organelles
        num_pc = 6
        if has_volume:
            df_pca["cell-volume"] = df_orga_perfolder.loc[df_orga_perfolder["organelle"].eq("ER"),"cell-volume"]
            columns = ["cell-volume",*organelles]
            num_pc = 7
        
        for orga in organelles:
            df_pca[orga] = df_orga_perfolder.loc[df_orga_perfolder["organelle"].eq(orga),property]
        
        if is_normalized:
            for col in columns:
                df_pca[col] = (df_pca[col]-df_pca[col].mean())/df_pca[col].std()

        df_pca.reset_index(inplace=True)
        df_pca["na_count"] = df_pca.isna().sum(axis=1)

        df_pca_washed = df_pca.fillna(0.)    
        print(folder,np.bincount(df_pca["na_count"]))

        np_pca = df_pca_washed[columns].to_numpy()
        pca = PCA(n_components=num_pc)
        pca.fit(np_pca)
        pca_components = [comp if comp[0]>0 else -comp for comp in pca.components_ ]

        for i_pc in range(len(pca_components)):
            base = pca_components[i_pc]
            df_pca_washed[f"proj{i_pc}"] = df_pca_washed.apply(lambda x:np.dot(base,x.loc[columns]),axis=1)

        pc2proj = []
        for k,proj in enumerate(pca_components):
            if len(pc2proj)>2:
                break
            if not ((num_pc==6 and np.argmax(np.abs(proj))==1) or (num_pc==7 and np.argmax(np.abs(proj))==2)):
                pc2proj.append(k)
        
        figproj = go.Figure()
        figcolors = ["purple","blue","green","yellow","orange","red"]
        for j,condi in enumerate(pd.unique(df_pca_washed["condition"])):
            figproj.add_trace(
                go.Scatter3d(
                    x=df_pca_washed.loc[df_pca_washed["condition"].eq(condi),f"proj{pc2proj[0]}"],
                    y=df_pca_washed.loc[df_pca_washed["condition"].eq(condi),f"proj{pc2proj[1]}"],
                    z=df_pca_washed.loc[df_pca_washed["condition"].eq(condi),f"proj{pc2proj[2]}"],
                    name=condi,
                    mode="markers",
                    marker=dict(size=2,color=figcolors[j],opacity=0.8)
                    
                )
            )
        figproj.update_layout(
            scene=dict(
                xaxis=dict(
                    title=f"proj{pc2proj[0]}",
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='grey',
                    showline = True,
                    zeroline=True,
                    zerolinecolor='black'
                ),
                yaxis=dict(
                    title=f"proj{pc2proj[1]}",
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='grey',
                    showline = True,
                    zeroline=True,
                    zerolinecolor='black'
                ),
                zaxis=dict(
                    title=f"proj{pc2proj[2]}",
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='grey',
                    showline = True,
                    zeroline=True,
                    zerolinecolor='black'
                )
            )
        )
        figproj.write_html(f"{folder_pca}/pca_projection3d_{folder}_{name}_pc{''.join([str(p) for p in pc2proj])}.html")

        fig_components = px.imshow(
            pca_components,
            x=columns, y=[f"PC{i}" for i in range(num_pc)],
            color_continuous_scale="RdBu_r", color_continuous_midpoint=0
        )
        fig_components.write_html(f"{folder_pca}/pca_components_{folder}_{name}.html")


    df_explained_variance_ratio = pd.DataFrame(dict_explained_variance_ratio)
    df_explained_variance_ratio.to_csv(f"{folder_pca}/explained_variance_ratio_{name}.csv",index=False)

    return None

for property in ["total-fraction","total","count"]:
    for has_cell in [True,False]:
        for if_normalized in [True,False]:
            make_pca_plots(property,has_volume=has_cell,is_normalized=if_normalized)
