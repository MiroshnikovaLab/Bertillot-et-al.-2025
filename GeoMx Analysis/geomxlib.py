import pandas as pd
import anndata as ad
import numpy as np 
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from adjustText import adjust_text

def barplot(adata, 
            x,
            labels,   
            colormap,
            bar_order = None,
            x_order = None,
            norm=True,
            figsize=(5,5),
            fontscale=2,
            style='white',
            rotation=0,
            kind='bar'
           ):
    
    
    grouped = pd.crosstab(adata.obs[x],
                        adata.obs[labels],
                        normalize=False)

    if norm:
        grouped = grouped.div(grouped.sum(axis=1), axis=0)*100


    if bar_order:
        bar_order = [col for col in bar_order if col in grouped.columns]
        grouped = grouped[bar_order]
        
    if x_order:
        grouped = grouped.loc[x_order, :]

    plt.rcParams["figure.figsize"] = figsize
    sns.set(font_scale = fontscale)
    sns.set_style(style)

    g = grouped.plot(kind=kind, 
                      stacked=True,
                      figsize=figsize,
                      width=0.8,
                      linewidth=0,
                      color = [colormap[col] for col in grouped.columns])
    
    for median in g.lines[:]:
        median.set_color('black')  # Change the color of the median line
        median.set_linewidth(1)  # Change

    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)

    
    plt.xticks(rotation=rotation, ha='center')
    
    plt.legend(loc='center left', 
               title=labels, 
               bbox_to_anchor=(1.0, 0.7), 
               prop={'size': 18}, 
               frameon=False)

    sns.despine()

    
    # return ax

def boxplot(df,
          x,
          y,
          order=None,
          figsize=(5,5),
          font_scale=2,
          xticks_rotation = 90,
          xlabel='',
          alpha=1,
          notch=False,
          showfliers=False,
        #   palette=sns.color_palette('tab10'),
          swarm_color='k',
          s=5,
          stripplot=True,
          ):

    plt.rcParams["figure.figsize"] = figsize
    sns.set(font_scale = font_scale)
    sns.set_style('white', rc={
    'xtick.bottom': True,
    'ytick.left': True,
        })

    g = sns.boxplot(df, 
                    x=x,
                    y=y, 
                    order=order,
                    notch=notch,
                #   medianprops={"color": "r", "linewidth": 2},
                    # palette=palette,
                    showfliers=showfliers,
                    # linewidth=0,
              color='w',
                boxprops=dict(linewidth=1, edgecolor='k'),  # Set edgecolor to black
                medianprops=dict(color='black', linewidth=1),
                    )
  
    for median in g.lines[:]:
        median.set_color('black')  # Change the color of the median line
        median.set_linewidth(1)  # Change
      
    if stripplot:

        g = sns.stripplot(df, 
                        x=x,
                        y=y, 
                        order=order,
                        palette=swarm_color,
                        alpha=alpha,
                        s=s
                        )

    g.tick_params(bottom=True)

    plt.xticks(rotation=xticks_rotation)
    plt.xlabel('')
    sns.despine()

def project_data_into_tissue(adata, 
                                patient_id,
                                color,
                                colormap,
                                fontsize=10
                                ):

    adata_filt = adata[adata.obs['patient_id']==patient_id]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10,10))
    
    texts = []

    scale_factor = 1
    
    for row in adata_filt.obs.itertuples():

        X = np.array([row.X + row.Width, row.X, row.X, row.X + row.Width, row.X + row.Width])*scale_factor
        Y = np.array([row.Y, row.Y, row.Y + row.Height, row.Y + row.Height, row.Y])*scale_factor

        row_dict = {field: getattr(row, field) for field in row._fields}
        ax.fill(X, Y, color=colormap[getattr(row, color)], alpha=1)

        # if row.region == 'CT':
        #     plt.plot(X, Y, '-', c='k', alpha=0.5)
        # else:
        #     plt.plot(X, Y, '--', c='k', alpha=0.5)

        # plt.plot(row.X + row.Width/2 , row.Y + row.Height/2 , 'ok')

        text = plt.text(
                        row.X*scale_factor+ row.Width/2,
                         row.Y*scale_factor+ row.Height/2,
                        #  row.X + 0.75* row.Width,
                        #  row.Y + 0.75*row.Height ,
                         f'{row.roi_id}', fontsize=15, 
                        fontweight='bold',
                        #  alpha=.8, 
                        color='white',
                         ha='center',
                           va='center'
                         )
        # texts.append(text)
        text.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])

        
    # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='k'),
                
    #                         avoid_self=True,
    #         force_text=(0.1, 0.4), 
    #         force_static=(0.1, 0.4), 
    #         force_pull=(0.01, 0.4), 
                
    #             )

    ax.invert_yaxis() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])   

    plt.title(patient_id, fontsize=30)

    # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.8)
    # cbar.set_label(marker, rotation=90, fontsize=25)
    # cbar.ax.tick_params(labelsize=25)

    plt.axis('equal')
    
def scanpy_preprocessing(adata, 
                         layer='raw',
                         n_neighbors=10,
                         n_pcs=30,
                         umap_random_state=42
                         ):
    
    adata_ = adata.copy()
    # adata_.layers['raw'] = adata_.X
    sc.pp.log1p(adata_, layer=layer)
    adata_.X = adata_.layers[layer]     
    sc.pp.pca(adata_, svd_solver='arpack')
    sc.pp.neighbors(adata_, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata_, random_state=umap_random_state)

    return adata_



def volcano_plot(df_, 
                 genes_list,
#                  name_list,
                 pval_threshold,
                 lfc_threshold,
                 fig_size=(7,5),
                 font_scale=1,
                 style='whitegrid'
                ):
    
    from adjustText import adjust_text
    from matplotlib import rcParams
    
    plt.rcParams["figure.figsize"] = fig_size
    sns.set(font_scale = font_scale)
    sns.set_style(style)
    
    # Non-significant genes
    ns = df_[(df_['pvals'] >= pval_threshold) | (abs(df_['lfc']) < lfc_threshold)]
    plt.scatter(ns['lfc'], ns['neg_log10_pval'], color='grey', alpha=0.2, label='Non-significant', s=10)
    
    sig_pos = df_[(df_['pvals'] < pval_threshold) & (df_['lfc'] >= lfc_threshold)]
    plt.scatter(sig_pos['lfc'], sig_pos['neg_log10_pval'], color='lightskyblue', label='Significant Up', s=10)
    
    sig_neg = df_[(df_['pvals'] < pval_threshold) & (df_['lfc'] <= -lfc_threshold)]
    plt.scatter(sig_neg['lfc'], sig_neg['neg_log10_pval'], color='lightskyblue', label='Significant Down', s=10)
    
    # Annotations for significant genes
    sig = df_[(df_['pvals'] < pval_threshold) & (abs(df_['lfc']) >= lfc_threshold)]
    
    neg_log10_pval_threshold = -np.log10(pval_threshold)
    
    texts = []
    sig_genes = []
    for _, row in sig.iterrows():
        if row['Gene'] in genes_list:
            sig_genes.append(row['Gene'])
            texts.append(plt.text(row['lfc'], row['neg_log10_pval'], row['Gene'], 
                                  fontsize=8, ha='right', va='bottom'))
            
    

    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
    
    # Add vertical line for fold change threshold
    plt.axvline(x=lfc_threshold, color='black', linestyle='--', lw=1)
    plt.axvline(x=-lfc_threshold, color='black', linestyle='--', lw=1)

    # Add horizontal line for p-value threshold
    plt.axhline(y=neg_log10_pval_threshold, color='black', linestyle='--', lw=1)

    plt.xlabel('Log2 Fold Change', fontsize=12)
    plt.ylabel('-Log10 p-value', fontsize=12)
#     plt.title(f"{name_list} {len(genes_list)} genes")

def extract_results_from_rank_genes(adata):
    adata_ = adata.copy()
    results = adata_.uns['rank_genes_groups']
    out = np.array([[0,0,0,0,0,0,0]])
    for group in results['names'].dtype.names:
        out = np.vstack((out, np.vstack((results['names'][group],
                                         results['scores'][group],
                                         results['pvals'][group],
                                         results['pvals_adj'][group],
                                          -np.log10(results['pvals'][group]),
                                         results['logfoldchanges'][group],
                                         np.array([group] * len(results['names'][group])).astype('object'))).T))
        
    df_dge = pd.DataFrame(out[1:], columns = ['Gene', 'scores', 'pvals', 'pval_adj','neg_log10_pval', 'lfc', 'group'])
    
    return df_dge

def convert_dict_values_to_uppercase(original_dict):
    uppercase_dict = {}

    for key, value in original_dict.items():
        uppercase_dict[key] = [item.upper() for item in value]
        
    return uppercase_dict




def compute_dge_from_pydeseq2(counts_df, 
                              metadata,
                              condition,
                              label,
                              min_count=2):
    
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats
    from pydeseq2.utils import load_example_data
    
    counts_df_ = counts_df.copy()
    metadata_ = metadata.copy()
    
    genes_to_keep = counts_df_.columns[counts_df_.sum(axis=0) >= min_count]
    counts_df_ = counts_df_[genes_to_keep]

    conditions = metadata_[condition] == label
    metadata_[condition] = conditions.astype(int).astype(str)
    
    inference = DefaultInference(n_cpus=6)
    
    dds = DeseqDataSet(
        counts=counts_df_.astype(int),
        metadata=metadata_,
        design_factors=[condition],
        refit_cooks=True,
        inference=inference,
    )

    dds.deseq2()
    dds.varm["LFC"]
    stat_res_1_vs_0 = DeseqStats(dds, contrast=[condition.replace('_', '-'), '1', '0'], inference=inference)
    stat_res_1_vs_0.summary()
    stat_res_1_vs_0.lfc_shrink(coeff=condition.replace('_', '-') + "_1_vs_0")
    df_dge = stat_res_1_vs_0.results_df
    df_dge = df_dge.reset_index()
    df_dge = df_dge.rename(columns={'log2FoldChange': "lfc", 
                                      "pvalue": "pvals",
                                      "padj": "pval_adj",
                                      "baseMean": "scores",
                                      "index": "Gene"
                                      })

    df_dge['neg_log10_pval'] = -np.log10(df_dge['pvals'])
    df_dge['group'] = [label]*len(df_dge)
    
    return df_dge




def compute_gsea(df_dges,
                gene_sets,
                conditions,
                pval_treshold=0.05,
                lfc_treshold=0,
                ):
    
    import gseapy as gp

    df_enr = []

    for cond in conditions:
        
        rows_to_keep = df_dges.group == cond
        degs_clust = df_dges[rows_to_keep]
        
        degs_sig = degs_clust[degs_clust.pval < pval_treshold]
        degs_up = degs_clust[degs_clust.lfc > lfc_treshold]
        degs_dw = degs_clust[degs_clust.lfc < -lfc_treshold]

        try:
            enr_up = gp.enrich(degs_up.Gene,
                            gene_sets=gene_sets,
                            outdir=None)
            
            enr_dw = gp.enrich(degs_dw.Gene,
                            gene_sets=gene_sets,
                            outdir=None)
            
            enr_up.res2d['Up_Down'] = "UP"
            enr_dw.res2d['Up_Down'] = "DOWN"
            enr_res = pd.concat([enr_up.res2d, enr_dw.res2d])
            enr_res['Condition'] = [cond]*len(enr_res)
            
            df_enr.append(enr_res)

        except:
            pass
        
    df_enr = pd.concat(df_enr)
    df_enr = df_enr.reset_index(drop=True)
    df_enr.loc[:, "log10 P-value"] = -np.log10(df_enr["Adjusted P-value"]).values


    return df_enr