import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.special import rel_entr
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from organelle_measure.data import read_results


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

# direction in which the growth rate grows.
extremes = {
    "EYrainbow_glucose":                    [0.,100.],
    "EYrainbow_glucose_largerBF":           [0.,100.],
    "EYrainbow_leucine_large":              [0.,100.],
    "EYrainbowWhi5Up_betaEstrodiol":        [10.,0.],
    "EYrainbow_rapamycin_1stTry":           [1000.,0.],
    "EYrainbow_rapamycin_CheckBistability": [300.,0.],
    "EYrainbow_1nmpp1_1st":                 [3000.,0.]
}

folder_i = Path("./data/results")
folder_o = Path("./data/figures")
folder_rate = Path("./data/growthrate")
folder_mutualinfo = Path("./data/mutual_information")
folder_correlation = Path("./data/correlation")
folder_pca_data = Path("./data/pca_data")
folder_pca_proj_extremes = Path("./data/pca_projection_extremes")
folder_pca_proj_all = Path("./data/pca_projection_all")
folder_pca_compare = Path("./data/pca_compare")

# READ FILES
df_bycell = read_results(folder_i,subfolders,(px_x,px_y,px_z))

# DATAFRAME FOR CORRELATION COEFFICIENT
# for folder in subfolders:
for folder in extremes.keys():
    pv_bycell = df_bycell[df_bycell['folder'].eq(folder)].set_index(['condition','field','idx-cell'])
    df_corrcoef = pd.DataFrame(index=pv_bycell.loc[pv_bycell["organelle"].eq("ER")].index)
    df_corrcoef.loc[:,'effective-length'] = np.sqrt(pv_bycell.loc[pv_bycell["organelle"].eq("ER"),'cell-area']/np.pi)
    df_corrcoef.loc[:,'cell-area'] = pv_bycell.loc[pv_bycell["organelle"].eq("ER"),'cell-area']
    df_corrcoef.loc[:,'cell-volume'] = pv_bycell.loc[pv_bycell["organelle"].eq("ER"),'cell-volume']
    properties = []
    for orga in [*organelles,"non-organelle"]:
        for prop in ['mean','count','total','total-fraction']:
            if (orga in ["ER","vacuole","non-organelle"]) and (prop in ["count","mean"]):
                continue
            prop_new = f"{prop}-{orga}"
            properties.append(prop_new)
            df_corrcoef.loc[:,prop_new] = pv_bycell.loc[pv_bycell["organelle"]==orga,prop]
    df_corrcoef.reset_index(inplace=True)

    # Kullback–Leibler_divergence of different conditions
    num_bin = 4
    organelles_kl = [f"total-fraction-{orga}" for orga in organelles]
    df_kldiverge = df_corrcoef[["condition",*organelles_kl]]
    # get grids in 6 dimensional space
    grids = np.array(list(map(
                lambda x:np.percentile(
                    df_kldiverge[x],
                    q=np.arange(0,100,100/num_bin)
                ),
                organelles_kl
            )))
    probabilities = {}
    for condi in np.sort(df_kldiverge["condition"].unique()):
        counts = np.array(list(map(
                    lambda x:np.digitize(
                        df_kldiverge.loc[df_kldiverge["condition"].eq(condi),x[0]],
                        x[1]
                    ),
                    zip(organelles_kl,grids)
                  ))) - 1
        indices = np.array(list(map(
                        lambda x: sum([xi*(num_bin+1)**i for xi,i in enumerate(x)]),
                        np.transpose(counts)
                  )))
        indices = np.bincount(indices)
        probs = np.zeros((num_bin+1)**len(organelles))
        probs[:len(indices)] = indices
        probs += 10**(-20)
        probs = probs/np.sum(probs)
        probabilities[condi] = probs
    normal = extremes[folder][-1]
    divergence = {}
    for condi in probabilities.keys():
        divergence[condi] = np.sum(rel_entr(probabilities[condi],probabilities[normal]))
    print(folder,":\n",divergence)

    # mutual information
    columns = ['condition','effective-length','cell-area','cell-volume',*properties]
    digitized = {}
    for column in columns:
        digitized[column] = np.digitize(
            df_corrcoef[column],
            bins=np.histogram(df_corrcoef[column],bins=100)[1]
        )
    num_col = len(columns)
    mx_mutualinfo = np.zeros((num_col,num_col))
    mx_miadjusted = np.zeros((num_col,num_col))
    mx_normalzied = np.zeros((num_col,num_col))
    for i,column0 in enumerate(columns):
        mx_mutualinfo[i,i] = metrics.mutual_info_score(digitized[column0],digitized[column0])
        mx_miadjusted[i,i] = 1.0
        mx_normalzied[i,i] = 1.0
        if (i+1)==num_col:
            break
        for j,column1 in enumerate(columns[i+1:]):
            mx_mutualinfo[i,i+j+1] = metrics.mutual_info_score(digitized[column0],digitized[column1])
            mx_miadjusted[i,i+j+1] = metrics.adjusted_mutual_info_score(digitized[column0],digitized[column1])
            mx_normalzied[i,i+j+1] = metrics.normalized_mutual_info_score(digitized[column0],digitized[column1])
    for matrix,mx_name in zip([mx_mutualinfo,mx_miadjusted,mx_normalzied],["mutual-info","adjusted-mutual-info","normalized-mutual-info"]):
        fig = px.imshow(
            matrix,
            x=columns,y=columns,
            color_continuous_scale="OrRd",range_color=[0,1]
        )
        fig.write_html(f"{folder_mutualinfo}/{mx_name}_{folder}.html")


    # Correlation coefficient
    np_corrcoef = df_corrcoef.loc[:,columns].to_numpy()
    corrcoef = np.corrcoef(np_corrcoef,rowvar=False)
    fig = px.imshow(
            corrcoef,
            x=columns,y=columns,
            color_continuous_scale = "RdBu_r",range_color=[-1,1]
        )
    fig.write_html(f"{folder_correlation}/corrcoef_{folder}.html")

    for condi in df_corrcoef["condition"].unique():
        np_corrcoef = df_corrcoef.loc[df_corrcoef['condition']==condi,['effective-length','cell-area','cell-volume',*properties]].to_numpy()
        corrcoef = np.corrcoef(np_corrcoef,rowvar=False)
        fig = px.imshow(
                corrcoef,
                x=columns,y=columns,
                color_continuous_scale="RdBu_r",range_color=[-1,1]
            )
        fig.write_html(f"{folder_correlation}/conditions/corrcoef_{folder}_{str(condi).replace('.','-')}.html")    

    # # Pairwise relation atlas
    # fig_pair = sns.PairGrid(df_corrcoef,hue="condition",vars=['effective-length','cell-area','cell-volume',*properties],height=3.0)
    # fig_pair.map_diag(sns.histplot)
    # # f_pairig.map_offdiag(sns.scatterplot)
    # fig_pair.map_upper(sns.scatterplot)
    # fig_pair.map_lower(sns.kdeplot)
    # fig_pair.add_legend()
    # fig_pair.savefig(f"{folder_correlation}/pairplot_{folder}.png")

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


def make_pca_plots(folder,property,groups=None,has_volume=False,is_normalized=False,non_organelle=False):
    name = f"{'all-conditions' if groups is None else 'extremes'}_{'has-cytoplasm' if non_organelle else 'no-cytoplasm'}_{'cell-volume' if has_volume else 'organelle-only'}_{property}_{'norm-mean-std' if is_normalized else 'raw'}"
        
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


    # Find the the direction of the condition change:
    df_centroid = df_pca.groupby("condition")[columns].mean()
    fitter_centroid = LinearRegression(fit_intercept=False)
    np_centroid = df_centroid.to_numpy()
    fitter_centroid.fit(np_centroid,np.ones(np_centroid.shape[0]))
    
    vec_centroid_start = np_centroid[0]
    vec_centroid_start[-1] = (1 - np.dot(fitter_centroid.coef_[:-1],vec_centroid_start[:-1]))/fitter_centroid.coef_[-1]
    vec_centroid_ended = np_centroid[-1]
    vec_centroid_ended[-1] = (1 - np.dot(fitter_centroid.coef_[:-1],vec_centroid_ended[:-1]))/fitter_centroid.coef_[-1]
    vec_centroid = vec_centroid_ended - vec_centroid_start
    vec_centroid = vec_centroid/np.linalg.norm(vec_centroid)

    # Get Principal Components (PCs)
    np_pca = df_pca[columns].to_numpy()
    pca = PCA(n_components=num_pc)
    pca.fit(np_pca)
    pca_components = pca.components_

    # Calculate cos<condition,PCs>, and realign the PCs
    cosine_pca = np.dot(pca_components,vec_centroid)
    for c in range(len(cosine_pca)):
        if cosine_pca[c] < 0:
            pca_components[c] = -pca_components[c]
    cosine_pca = np.abs(cosine_pca)
    # save and plot PCs without sorting.
    np.savetxt(f"{folder_pca_data}/pca-components_{folder}_{name}.txt",pca_components)    
    fig_components = px.imshow(
        pca_components,
        x=columns, y=[f"PC{i}" for i in range(num_pc)],
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0
    )
    fig_components.write_html(f"{folder_pca_data}/pca_components_{folder}_{name}.html")

    # Sort PCs according to the cosine
    arg_cosine = np.argsort(cosine_pca)[::-1]
    pca_components_sorted = pca_components[arg_cosine]
    # save and plot the PCs with sorting
    np.savetxt(f"{folder_pca_compare}/condition-sorted-pca-components_{folder}_{name}.txt",pca_components_sorted)
    fig_components_sorted = px.imshow(
        pca_components_sorted,
        x=columns, y=[f"PC{i}" for i in arg_cosine],
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0
    )
    fig_components_sorted.write_html(f"{folder_pca_compare}/condition-sorted-pca_components_{folder}_{name}.html")
    # plot the cosine 
    plt.figure()
    plt.barh(np.arange(num_pc),cosine_pca[arg_cosine[::-1]],align='center')
    plt.yticks(np.arange(num_pc),[f"PC{i}" for i in arg_cosine[::-1]])
    plt.xlabel(r"$cos\left<condition\ vector,PC\right>$")
    plt.title(f"{folder}")
    plt.savefig(f"{folder_pca_compare}/condition-sorted-cosine_{folder}_{name}.png")


    # Draw projections onto the PCs
    for i_pc in range(len(pca_components)):
        base = pca_components[i_pc]
        df_pca[f"proj{i_pc}"] = df_pca.apply(lambda x:np.dot(base,x.loc[columns]),axis=1)
    
    pc2proj = arg_cosine[:3]

    df_pca_extremes = df_pca.loc[df_pca["condition"].isin(groups)]
    figproj = plt.figure(figsize=(10,8))
    ax = figproj.add_subplot(projection="3d")
    for condi in pd.unique(df_pca_extremes["condition"]):
        pc_x = df_pca_extremes.loc[df_pca_extremes["condition"].eq(condi),f"proj{pc2proj[0]}"],
        pc_y = df_pca_extremes.loc[df_pca_extremes["condition"].eq(condi),f"proj{pc2proj[1]}"],
        pc_z = df_pca_extremes.loc[df_pca_extremes["condition"].eq(condi),f"proj{pc2proj[2]}"],
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
    ax.set_xlim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[0]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
    ax.set_ylim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[1]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
    ax.set_zlim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[2]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
    ax.legend(loc=(1.04,0.5))
    figproj.savefig(f"{folder_pca_proj_extremes}/pca_projection3d_{folder}_{name}_pc{''.join([str(p) for p in pc2proj])}.png")
    sns.set_style("whitegrid")
    # 2d projections
    for first,second in ((0,1),(0,2),(1,2)):
        plt.figure()
        sns.scatterplot(
            data=df_pca_extremes,
            x=f"proj{pc2proj[first]}",y=f"proj{pc2proj[second]}",
            hue="condition",palette="tab10",alpha=0.5
        )
        plt.savefig(f"{folder_pca_proj_extremes}/pca_projection2d_{folder}_{name}_pc{pc2proj[first]}{pc2proj[second]}.png")
    
    # draw with Plotly:
    figproj = go.Figure()
    for condi in pd.unique(df_pca["condition"]):
        figproj.add_trace(
            go.Scatter3d(
                x=df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[0]}"],
                y=df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[1]}"],
                z=df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[2]}"],
                name=condi,
                mode="markers",
                marker=dict(
                            size=2,
                            # color=figcolors[j],
                            opacity=0.8
                       )   
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
                showline=True,
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
    figproj.write_html(f"{folder_pca_proj_all}/pca_projection3d_{folder}_{name}_pc{''.join([str(p) for p in pc2proj])}.html")

    return None

# for property in ["total-fraction","total","count"]:
#     for has_cell in [True,False]:
#         for if_normalized in [True,False]:
#             make_pca_plots(property,has_volume=has_cell,is_normalized=if_normalized)


# dict_explained_variance_ratio = {}
for folder in extremes.keys():
    make_pca_plots(folder,"total-fraction",groups=extremes[folder],has_volume=True,is_normalized=True,non_organelle=False)
# df_explained_variance_ratio = pd.DataFrame(dict_explained_variance_ratio)
# df_explained_variance_ratio.to_csv(f"{folder_pca_data}/explained_variance_ratio_{name}.csv",index=False)

df_trivial = pd.concat(
    [
        df_bycell.loc[df_bycell["folder"].eq(ex) & df_bycell["condition"].eq(extremes[ex][-1])] 
        for ex in extremes.keys()
    ],
    ignore_index=True
)