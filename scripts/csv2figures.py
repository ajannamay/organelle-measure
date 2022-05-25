import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Global Variables

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
folder_o = Path("./data/figures")
folder_rate = Path("./data/growthrate")
# folder_pca = Path("./data/figures/pca_transparent_bkgd")


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
    [(*index,orga) for index in pivot_cell_bycell.index.to_list() for orga in [*organelles,'non-organelle']],
    names=['folder','condition','field','idx-cell','organelle']
)
pivot_bycell = pd.DataFrame(index=index_bycell)
# combine data with index
pivot_bycell.loc[pivot_orga_bycell_mean.index,"mean"] = pivot_orga_bycell_mean
pivot_bycell.loc[pivot_orga_bycell_mean.index,"count"] = pivot_orga_bycell_nums
pivot_bycell.loc[pivot_orga_bycell_mean.index,"total"] = pivot_orga_bycell_totl
pivot_bycell.fillna(0.,inplace=True)

# include cell data
pivot_bycell.reset_index("organelle",inplace=True) # comment out after 1st run 
pivot_bycell.loc[:,"cell-area"] = pivot_cell_bycell.loc[:,"area"]
pivot_bycell.loc[:,"cell-volume"] = pivot_cell_bycell.loc[:,"effective-volume"]
pivot_bycell.loc[:,"total-fraction"] = pivot_bycell.loc[:,"total"]/pivot_bycell.loc[:,"cell-volume"]

# calculate properties of regions that are not organelles
df_bycell = pivot_bycell.reset_index()

df_none = df_bycell[df_bycell['organelle'].ne("non-organelle")].groupby(['folder','condition','field','idx-cell'])[['total','cell-volume','total-fraction']].agg({'total':'sum','cell-volume':'first','total-fraction':'sum'})
pivot_bycell.loc[pivot_bycell['organelle'].eq("non-organelle"),"count"] = 1
pivot_bycell.loc[pivot_bycell['organelle'].eq("non-organelle"),"mean"] = df_none["cell-volume"] - df_none["total"]
pivot_bycell.loc[pivot_bycell['organelle'].eq("non-organelle"),"total"] = df_none["cell-volume"] - df_none["total"]
pivot_bycell.loc[pivot_bycell['organelle'].eq("non-organelle"),"total-fraction"] = 1 - df_none["total-fraction"]

df_bycell = pivot_bycell.reset_index()

# DATAFRAME FOR CORRELATION COEFFICIENT
for folder in subfolders:
    pv_bycell = df_bycell[df_bycell['folder'].eq(folder)].set_index(['condition','field','idx-cell'])
    df_corrcoef = pd.DataFrame(index=pivot_cell_bycell.loc[folder,:].index)
    df_corrcoef.loc[:,'effective-length'] = np.sqrt(pivot_cell_bycell.loc[folder,'area']/np.pi)
    df_corrcoef.loc[:,'cell-area'] = pivot_cell_bycell.loc[folder,'area']
    df_corrcoef.loc[:,'cell-volume'] = pivot_cell_bycell.loc[folder,'effective-volume']
    properties = []
    for orga in [*organelles,"non-organelle"]:
        for prop in ['mean','count','total','total-fraction']:
            if (orga in ["ER","vacuole","non-organelle"]) and (prop in ["count","mean"]):
                continue
            prop_new = f"{prop}-{orga}"
            properties.append(prop_new)
            df_corrcoef.loc[:,prop_new] = pv_bycell.loc[pv_bycell["organelle"]==orga,prop]
    df_corrcoef.reset_index(inplace=True)

    # Correlation coefficient
    np_corrcoef = df_corrcoef.loc[:,['condition','effective-length','cell-area','cell-volume',*properties]].to_numpy()
    corrcoef = np.corrcoef(np_corrcoef,rowvar=False)
    fig = px.imshow(
            corrcoef,
            x=['condition','effective-length','cell-area','cell-volume',*properties],
            y=['condition','effective-length','cell-area','cell-volume',*properties],
            color_continuous_scale = "RdBu_r",range_color=[-1,1]
        )
    fig.write_html(f"{folder_o}/corrcoef-number_{folder}.html")

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

    # # Pairwise relation atlas
    # fig_pair = sns.PairGrid(df_corrcoef,hue="condition",vars=['effective-length','cell-area','cell-volume',*properties],height=3.0)
    # fig_pair.map_diag(sns.histplot)
    # # f_pairig.map_offdiag(sns.scatterplot)
    # fig_pair.map_upper(sns.scatterplot)
    # fig_pair.map_lower(sns.kdeplot)
    # fig_pair.add_legend()
    # fig_pair.savefig(f"{folder_o}/pairplot_{folder}.png")

    # # Pairwise relation individual
    # sns.set_palette(sns.blend_palette(['red','blue']))
    # g = sns.jointplot(
    #                   data=df_corrcoef, 
    #                   x="cell-volume", y="total-fraction-mitochondria",
    #                   hue="condition", kind="kde"
    #     )
    # # g.plot_joint(sns.kdeplot, zorder=0, levels=6)
    # g.plot_marginals(sns.rugplot, height=-.15, clip_on=False)


# EXPERIMENT CONDITION LEVEL

pivot_bycondition = df_bycell.groupby(['folder','organelle','condition']).mean()[['mean','count','total','cell-area','cell-volume','total-fraction']]
pivot_bycondition["cell_count"] = df_bycell[['folder','organelle','condition','mean']].groupby(['folder','organelle','condition']).count()
df_bycondition = pivot_bycondition.reset_index()
df_rates = pd.read_csv(str(folder_rate/"growth_rate.csv"))
df_rates.rename(columns={"experiment":"folder"},inplace=True)
df_bycondition.set_index(["folder","condition"],inplace=True)
df_rates.set_index(["folder","condition"],inplace=True)
df_bycondition.loc[:,"growth_rate"] = df_rates["growth_rate"]
df_bycondition.reset_index(inplace=True)
fig = px.line(
    df_bycondition.loc[df_bycondition['organelle'].eq("non-organelle")],
    x='growth_rate',y='total',
    color="folder"
)
fig.write_html(str(folder_rate/"non-organelle-vol-total_growth-rate.html"))
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

def make_pca_plots(folder,property,groups=None,has_volume=False,is_normalized=False,non_organelle=False):
    name = f"{'all-conditions' if groups is None else 'extremes'}_{'has-cytoplasm' if non_organelle else 'no-cytoplasm'}_{'cell-volume' if has_volume else 'organelle-only'}_{property}_{'norm-mean-std' if is_normalized else 'raw'}"
    dict_explained_variance_ratio = {}
        
    df_orga_perfolder = df_bycell[df_bycell["folder"].eq(folder)]
    df_orga_perfolder.set_index(["condition","field","idx-cell"],inplace=True)
    idx = df_orga_perfolder.groupby(["condition","field","idx-cell"]).count().index
    
    columns = [*organelles,"non-organelle"] if non_organelle else organelles
    df_pca = pd.DataFrame(index=idx,columns=columns)
    num_pc = 7 if non_organelle else 6
    
    for orga in columns:
        df_pca[orga] = df_orga_perfolder.loc[df_orga_perfolder["organelle"].eq(orga),property]
    
    if has_volume:
        df_pca["cell-volume"] = df_orga_perfolder.loc[df_orga_perfolder["organelle"].eq("ER"),"cell-volume"]
        columns = ["cell-volume",*columns]
        num_pc += 1
    
    if is_normalized:
        for col in columns:
            df_pca[col] = (df_pca[col]-df_pca[col].mean())/df_pca[col].std()
    
    df_pca.reset_index(inplace=True)
    
    # # np.nan already filled before doing PCA.
    # df_pca["na_count"] = df_pca.isna().sum(axis=1)
    # df_pca_washed = df_pca.fillna(0.)    
    # print(folder,np.bincount(df_pca["na_count"]))
    
    np_pca = df_pca[columns].to_numpy()
    pca = PCA(n_components=num_pc)
    pca.fit(np_pca)
    pca_components = np.array([comp if comp[np.argmax(np.abs(comp))]>0 else -comp for comp in pca.components_])
    np.savetxt(f"{folder_o}/pca-components_{folder}_{name}.txt",pca_components)
    
    for i_pc in range(len(pca_components)):
        base = pca_components[i_pc]
        df_pca[f"proj{i_pc}"] = df_pca.apply(lambda x:np.dot(base,x.loc[columns]),axis=1)
    pc2proj = []
    
    for k,proj in enumerate(pca_components):
        if len(pc2proj)>2:
            break
        if np.argmax(np.abs(proj))!=columns.index("vacuole"):
            pc2proj.append(k)

    if groups is not None:
        df_pca = df_pca.loc[df_pca["condition"].isin(groups)]

    figproj = plt.figure(figsize=(10,8))
    ax = figproj.add_subplot(projection="3d")
    for condi in pd.unique(df_pca["condition"]):
        pc_x = df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[0]}"],
        pc_y = df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[1]}"],
        pc_z = df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[2]}"],
        ax.scatter(
            pc_x, pc_y, pc_z,
            alpha=0.2,label=f"{condi}"
        )
    ax.set_xlabel(f"proj {pc2proj[0]}")
    ax.set_ylabel(f"proj {pc2proj[1]}")
    ax.set_zlabel(f"proj {pc2proj[2]}")
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlim(*(np.percentile(df_pca[f"proj{pc2proj[0]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
    ax.set_ylim(*(np.percentile(df_pca[f"proj{pc2proj[1]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
    ax.set_zlim(*(np.percentile(df_pca[f"proj{pc2proj[2]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
    ax.legend(loc=(1.04,0.5))
    figproj.savefig(f"{folder_o}/pca_projection3d_{folder}_{name}_pc{''.join([str(p) for p in pc2proj])}.png")
    sns.set_style("whitegrid")
    plt.figure()
    sns.scatterplot(
        data=df_pca,
        x=f"proj{pc2proj[0]}",y=f"proj{pc2proj[1]}",
        hue="condition",palette="tab10",alpha=0.5
    )
    plt.savefig(f"{folder_o}/pca_projection2d_{folder}_{name}_pc{pc2proj[0]}{pc2proj[1]}.png")
    plt.figure()
    sns.scatterplot(
        data=df_pca,
        x=f"proj{pc2proj[0]}",y=f"proj{pc2proj[2]}",
        hue="condition",palette="tab10",alpha=0.5
    )
    plt.savefig(f"{folder_o}/pca_projection2d_{folder}_{name}_pc{pc2proj[0]}{pc2proj[2]}.png")
    plt.figure()
    sns.scatterplot(
        data=df_pca,
        x=f"proj{pc2proj[1]}",y=f"proj{pc2proj[2]}",
        hue="condition",palette="tab10",alpha=0.5
    )
    plt.savefig(f"{folder_o}/pca_projection2d_{folder}_{name}_pc{pc2proj[1]}{pc2proj[2]}.png")
    
    # # draw with Plotly:
    # figproj = go.Figure()
    # for condi in pd.unique(df_pca["condition"]):
    #     figproj.add_trace(
    #         go.Scatter3d(
    #             x=df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[0]}"],
    #             y=df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[1]}"],
    #             z=df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[2]}"],
    #             name=condi,
    #             mode="markers",
    #             marker=dict(
    #                         size=2,
    #                         # color=figcolors[j],
    #                         opacity=0.8
    #                    )   
    #         )
    #     )
    # figproj.update_layout(
    #     scene=dict(
    #         xaxis=dict(
    #             title=f"proj{pc2proj[0]}",
    #             backgroundcolor='rgba(0,0,0,0)',
    #             gridcolor='grey',
    #             showline = True,
    #             zeroline=True,
    #             zerolinecolor='black'
    #         ),
    #         yaxis=dict(
    #             title=f"proj{pc2proj[1]}",
    #             backgroundcolor='rgba(0,0,0,0)',
    #             gridcolor='grey',
    #             showline=True,
    #             zeroline=True,
    #             zerolinecolor='black'
    #         ),
    #         zaxis=dict(
    #             title=f"proj{pc2proj[2]}",
    #             backgroundcolor='rgba(0,0,0,0)',
    #             gridcolor='grey',
    #             showline = True,
    #             zeroline=True,
    #             zerolinecolor='black'
    #         )
    #     )
    # )
    # figproj.write_html(f"{folder_o}/pca_projection3d_{folder}_{name}_pc{''.join([str(p) for p in pc2proj])}.html")
    fig_components = px.imshow(
        pca_components,
        x=columns, y=[f"PC{i}" for i in range(num_pc)],
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0
    )
    fig_components.write_html(f"{folder_o}/pca_components_{folder}_{name}.html")


    df_explained_variance_ratio = pd.DataFrame(dict_explained_variance_ratio)
    df_explained_variance_ratio.to_csv(f"{folder_o}/explained_variance_ratio_{name}.csv",index=False)

    return None

# for property in ["total-fraction","total","count"]:
#     for has_cell in [True,False]:
#         for if_normalized in [True,False]:
#             make_pca_plots(property,has_volume=has_cell,is_normalized=if_normalized)

extremes = {
    "EYrainbow_glucose_largerBF":    [0.,100.],
    "EYrainbow_leucine_large":       [0.,100.],
    "EYrainbowWhi5Up_betaEstrodiol": [0.,10.],
    "EYrainbow_rapamycin_1stTry":    [0.,1000.],
    "EYrainbow_1nmpp1_1st":          [0.,3000.]
}
for folder in extremes.keys():
    make_pca_plots(folder,"total-fraction",groups=extremes[folder],has_volume=True,is_normalized=True,non_organelle=False)

# Radar charts for the principal components of PCA
df_pc_components = []
for folder in extremes.keys():
    np_pc = np.loadtxt(str(folder_o/f"pca-components_{folder}_organellle-only_total-fraction_raw.txt"))
    for i,pc in enumerate(np_pc):
        if np.max(pc)!=np.max(np.abs(pc)):
            pc = -pc
        df_pc_components.append({
            "experiment": folder,
            "PC": f"PC{i}",
            "organelle": organelles,
            "value": pc,
            "abs":   np.abs(pc),
            "argmax": np.argmax(np.abs(pc))
        })
df_pc_components = pd.concat([pd.DataFrame(d) for d in df_pc_components],ignore_index=True)

# compare the i-th most significant principal components in different experiments 
for i in range(6):
    plt.figure(figsize=(8,4))
    ax = sns.barplot(
        data=df_pc_components.loc[df_pc_components["PC"].eq(f"PC{i}")],
        x="organelle",y="value",hue="experiment",
        palette="Set2"
    )
    sns.move_legend(ax, "upper right", bbox_to_anchor=(1.75, 1))
    plt.title(f"PC{i}")
    plt.subplots_adjust(right=0.65)
    plt.savefig(f"{folder_o}/PC_compare_pc{i}.png")

# compare the principal components that j-th organelle stands out
for i in range(6):
    plt.figure(figsize=(8,4))
    ax = sns.barplot(
        data=df_pc_components.loc[df_pc_components["argmax"].eq(i)],
        x="organelle",y="value",hue="experiment",
        palette="Set2"
    )
    sns.move_legend(ax, "upper right", bbox_to_anchor=(1.75, 1))
    plt.title(f"{organelles[i]}")
    plt.subplots_adjust(right=0.65)
    plt.savefig(f"{folder_o}/PC_compare_organelle_{organelles[i]}.png")
    