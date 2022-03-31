# -*- coding: utf-8 -*-
"""
@author: mianfg
"""

from scipy.cluster import hierarchy
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from collections import Counter as counter

def plot_heatmap(instance, show=True, save_route=None):
    sns.set()
    ax = sns.heatmap(instance['centroids'], cmap='YlGnBu', annot=instance['centroids'], fmt='.3f')
    ax.set(xlabel='Variables', ylabel='Cluster')
    #b, t = ax.get_ylim()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    if save_route: plt.savefig(save_route)
    if show: plt.show()
    
    plt.clf()

def plot_cluster_sizes(instance, show=True, save_route=None):
    count = dict(counter(instance['results']))

    sns.set()
    plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots()
    ax.bar(count.keys(), count.values(), align='center', alpha=0.5)
    ax.set(xlabel='Índice de clúster', ylabel='Número de elementos')
    
    if save_route: fig.savefig(save_route, transparent=False, dpi=80, bbox_inches="tight")
    if show: plt.show()

def plot_scatter_matrix(instance, show=True, save_route=None):
    sns.set()
    vs = list(instance['X_clusters'])
    vs.remove('cluster')
    ax = sns.pairplot(instance['X_clusters'], vars=vs, hue='cluster', palette='Paired', plot_kws={'s': 25}, diag_kind='hist')
    plt.subplots_adjust(wspace=.03, hspace=.03)
    
    if save_route: plt.savefig(save_route)
    if show: plt.show()
    
    plt.clf()

def plot_dist(instance, show=True, save_route=None):
    columns = list(instance['X'].columns)
    n_var = len(columns)

    sns.set()
    fig, axes = plt.subplots(instance['num_clusters'], n_var, sharey=True, figsize=(15,15))
    fig.subplots_adjust(wspace=0.007, hspace = 0.04)

    colors = sns.color_palette(palette=None, n_colors=instance['num_clusters'], desat=None)

    rango = [] # contendrá el mín y el máx para cada variable de todos los clusters
    for j in range(n_var):
        rango.append([instance['X_clusters'][columns[j]].min(), instance['X_clusters'][columns[j]].max()])
    
    for i in range(instance['num_clusters']):
        dat_filt = instance['X_clusters'].loc[instance['X_clusters']['cluster']==i]
        for j in range(n_var):
            ax = sns.distplot(dat_filt[columns[j]], color = colors[i], label = "", ax = axes[i,j])
            if (i==instance['num_clusters']-1):
                axes[i,j].set_xlabel(columns[j])
            else:
                axes[i,j].set_xlabel("")
        
            if (j==0):
                axes[i,j].set_ylabel("Cluster "+str(i))
            else:
                axes[i,j].set_ylabel("")
       
            axes[i,j].set_yticks([])
            axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            axes[i,j].grid(axis='y', b=False)
       
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]), rango[j][1]+0.05*(rango[j][1]-rango[j][0]))
            
    if save_route: plt.savefig(save_route)
    if show: plt.show()
    
    plt.clf()

def plot_hist(instance, show=True, save_route=None):
    columns = list(instance['X'].columns)
    n_var = len(columns)

    sns.set()
    fig, axes = plt.subplots(instance['num_clusters'], n_var, sharey=True, figsize=(15,15))
    fig.subplots_adjust(wspace=0.007, hspace = 0.04)

    colors = sns.color_palette(palette=None, n_colors=instance['num_clusters'], desat=None)

    rango = [] # contendrá el mín y el máx para cada variable de todos los clusters
    for j in range(n_var):
        rango.append([instance['X_clusters'][columns[j]].min(), instance['X_clusters'][columns[j]].max()])
    
    for i in range(instance['num_clusters']):
        dat_filt = instance['X_clusters'].loc[instance['X_clusters']['cluster']==i]
        for j in range(n_var):
            ax = sns.histplot(x=dat_filt[columns[j]], color = colors[i], label = "", ax = axes[i,j], kde=True)
            if (i==instance['num_clusters']-1):
                axes[i,j].set_xlabel(columns[j])
            else:
                axes[i,j].set_xlabel("")
        
            if (j==0):
                axes[i,j].set_ylabel("Cluster "+str(i))
            else:
                axes[i,j].set_ylabel("")
       
            axes[i,j].set_yticks([])
            axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            axes[i,j].grid(axis='y', b=False)
       
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]), rango[j][1]+0.05*(rango[j][1]-rango[j][0]))
            
    if save_route: plt.savefig(save_route)
    if show: plt.show()
    
    plt.clf()

def plot_boxplot(instance, show=True, save_route=None):
    columns = list(instance['X'].columns)
    n_var = len(columns)

    sns.set()
    fig, axes = plt.subplots(instance['num_clusters'], n_var, sharey=True, figsize=(17, 17))
    fig.subplots_adjust(wspace=0.04, hspace=0.5)
    
    colors = sns.color_palette(palette=None, n_colors=instance['num_clusters'], desat=None)

    rango = [] # contendrá el mín y el máx para cada variable de todos los clusters
    for j in range(n_var):
        rango.append([instance['X_clusters'][columns[j]].min(), instance['X_clusters'][columns[j]].max()])

    for i in range(instance['num_clusters']):
        dat_filt =  instance['X_clusters'].loc[instance['X_clusters']['cluster']==i]
        for j in range(n_var):
            ax = sns.boxplot(x=dat_filt[columns[j]], color=colors[i], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])

            if (i==instance['num_clusters']-1):
                axes[i,j].set_xlabel(columns[j])
            else:
                axes[i,j].set_xlabel("")
            
            if (j==0):
                axes[i,j].set_ylabel("Cluster "+str(i))
            else:
                axes[i,j].set_ylabel("")
            
            axes[i,j].set_yticks([])
            axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            axes[i,j].grid(axis='y', b=False)
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]), rango[j][1]+0.05*(rango[j][1]-rango[j][0]))

    if save_route: fig.savefig(save_route)
    if show: plt.show()

    plt.clf()

def plot_kde(instance, show=True, save_route=None):
    columns = list(instance['X'].columns)
    n_var = len(columns)

    sns.set()
    fig, axes = plt.subplots(instance['num_clusters'], n_var, sharey=True, figsize=(15,15))
    fig.subplots_adjust(wspace=0.007, hspace = 0.04)

    colors = sns.color_palette(palette=None, n_colors=instance['num_clusters'], desat=None)

    rango = [] # contendrá el mín y el máx para cada variable de todos los clusters
    for j in range(n_var):
        rango.append([instance['X_clusters'][columns[j]].min(), instance['X_clusters'][columns[j]].max()])
    
    for i in range(instance['num_clusters']):
        dat_filt = instance['X_clusters'].loc[instance['X_clusters']['cluster']==i]
        for j in range(n_var):
            ax = sns.kdeplot(x=dat_filt[columns[j]], shade = True, color = colors[i], label = "", ax = axes[i,j])
            if (i==instance['num_clusters']-1):
                axes[i,j].set_xlabel(columns[j])
            else:
                axes[i,j].set_xlabel("")
       
            if (j==0):
                axes[i,j].set_ylabel("Cluster "+str(i))
            else:
                axes[i,j].set_ylabel("")
       
            axes[i,j].set_yticks([])
            axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            axes[i,j].grid(axis='y', b=False)
       
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]), rango[j][1]+0.05*(rango[j][1]-rango[j][0]))
            
    if save_route: plt.savefig(save_route)
    if show: plt.show()
    
    plt.clf()

"""
def plot_inter(instance, show=True, save_route=None):
    fig = plt.figure()
    mpl.stype.use('default')
    mds = MDS(random_state=7)
    centers_mds = mds.fit_transform(instance['centroids'])
    plt.scatter(centers_mds[:,0], centers_mds[:,1], s=size**1.6, alpha=0.75, c=colors)
    for i in range(instance['num_clusters']):
        plt.annotate(str(i+1), xy=centers_mds[i], fontsize=18, va='center', ha='center')
    xl, xr = plt.xlim()
    yl, yr = plt.ylim()
    plt.xlim(xl-(xr-xl)*0.13,xr+(xr-xl)*0.13)
    plt.ylim(yl-(yr-yl)*0.13,yr+(yr-yl)*0.13)
    plt.xticks([])
    plt.yticks([])
    fig.set_size_inches(15,15)

    if save_route: plt.savefig(save_route)
    if show: plt.show()
    
    plt.clf()
"""

def plot_dendrogram(instance, p=None, show=True, save_route=None):
    if not instance['ward_specific']:
        raise ValueError('Must have executed ward_specific first')
        
    plt.figure(1)
    plt.clf()
    hierarchy.dendrogram(instance['ward_specific']['linkage_array'], orientation="left", p=p, truncate_mode='lastp' if p else None)
    if save_route: plt.savefig(save_route)
    if show: plt.show()

    plt.clf()

def plot_dendrogram_heat(instance, show=True, save_route=None):
    if not instance['ward_specific']:
        raise ValueError('Must have executed ward_specific first')
    
    columns = list(instance['X'].columns)
    
    X_filtered_norm_df = pd.DataFrame(instance['ward_specific']['X_filtered_norm'],
                                      index=instance['ward_specific']['X_filtered'].index, columns=columns)
    
    sns.set()
    ax = sns.clustermap(X_filtered_norm_df, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)

    if save_route: ax.savefig(save_route)
    if show: plt.show()

    plt.clf()
