#!/usr/bin/env python
# coding: utf-8

# # `IN` Práctica 2. Análisis relacional mediante segmentación

# Miguel Ángel Fernández Gutiérrez <<mianfg@correo.ugr.es>>

# ## Caso 2. Renta, transferencias e impuestos

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

from sklearn import preprocessing as sk_preprocessing
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS
from math import floor

import cluster, visualization, common
from cluster import ClusterAlgorithm

mpl.rcParams['figure.dpi'] = 120


# Leemos el _dataset_, los usaremos en todos los casos:

# In[2]:


data = pd.read_csv('./data/datos_hogar_2020.csv')


# **Filtrado:** ninguno

# In[3]:


case = data.copy()


# **Selección de variables** y renombrado

# In[4]:


case_columns = {
    'HY020': 'renta_disponible',
    'HY080N': 'transferencias_percibidas',
    'HY130N': 'transferencias_abonadas',
    'HY140G': 'impuesto_renta'
}
case = case.rename(columns=case_columns)
case = case[case_columns.values()]
case


# **Valores perdidos:** no hay valores perdidos

# In[5]:


case.isna().sum()


# **Valores nulos:** hay muchos valores que son 0 en `transferencias_percibidas` y `transferencias_abonadas`
# 
# _**Nota:** a continuación se muestran, fijada la variable, el número de instancias que tienen su valor no nulo._

# In[6]:


case.astype(bool).sum(axis=0)


# **Instanciamos y ejecutamos los algoritmos**

# In[13]:


# nota: en Birch hemos tenido que decrementar el threshold
algorithms = [
    ClusterAlgorithm(KMeans, name='K-Means', init='k-means++', n_clusters=5, n_init=5, random_state=common.RANDOM_SEED, centroid_attr="cluster_centers_"),
    ClusterAlgorithm(Birch, name='Birch', branching_factor=25, threshold=0.08, n_clusters=5, centroid_attr="subcluster_centers_"),
    ClusterAlgorithm(DBSCAN, name='DBSCAN', eps=0.15, min_samples=5),
    ClusterAlgorithm(MeanShift, name='MeanShift', centroid_attr="cluster_centers_"),
    ClusterAlgorithm(AgglomerativeClustering, name='Ward', n_clusters=5, linkage='ward')
]


# In[14]:


print("Algoritmos a usar:\n")

for algorithm in algorithms:
    print(algorithm.__repr__())


# In[15]:


print("Ejecutando instancias...\n")

for algorithm in algorithms:
    algorithm.run_instances(case, verbose=True)


# In[16]:


print("Calculando métricas:\n")

for algorithm in algorithms:
    algorithm.calculate_metrics(cluster.metrics, verbose=True)


# In[17]:


case_metrics = pd.DataFrame(columns=cluster.metrics.keys())
for algorithm in algorithms:
    case_metrics.loc[algorithm.algorithm_name] = algorithm.instances[0]['metrics']
case_metrics


# In[68]:


print(case_metrics.to_latex())


# ### Específico de Ward

# In[18]:


cluster.ward_specific(algorithms[4].instances[0])


# ### Visualización

# #### Heatmaps

# ##### K-Means

# In[19]:


visualization.plot_heatmap(algorithms[0].instances[0])


# ##### Birch

# In[20]:


visualization.plot_heatmap(algorithms[1].instances[0])


# ##### DBSCAN

# In[21]:


visualization.plot_heatmap(algorithms[2].instances[0])


# ##### Mean Shift

# In[22]:


visualization.plot_heatmap(algorithms[3].instances[0])


# ##### Ward

# In[23]:


visualization.plot_heatmap(algorithms[4].instances[0])


# #### Cluster sizes

# ##### K-Means

# In[24]:


visualization.plot_cluster_sizes(algorithms[0].instances[0])


# ##### Birch

# In[25]:


visualization.plot_cluster_sizes(algorithms[1].instances[0])


# ##### DBSCAN

# In[26]:


visualization.plot_cluster_sizes(algorithms[2].instances[0])


# ##### Mean Shift

# In[27]:


visualization.plot_cluster_sizes(algorithms[3].instances[0])


# ##### Ward

# In[28]:


visualization.plot_cluster_sizes(algorithms[4].instances[0])


# #### Scatter matrix

# ##### K-Means

# In[29]:


visualization.plot_scatter_matrix(algorithms[0].instances[0])


# ##### Birch

# In[30]:


visualization.plot_scatter_matrix(algorithms[1].instances[0])


# ##### DBSCAN

# In[31]:


visualization.plot_scatter_matrix(algorithms[2].instances[0])


# ##### Mean Shift

# In[32]:


visualization.plot_scatter_matrix(algorithms[3].instances[0])


# ##### Ward

# In[33]:


visualization.plot_scatter_matrix(algorithms[4].instances[0])


# #### Boxplot

# ##### K-Means

# In[34]:


visualization.plot_boxplot(algorithms[0].instances[0])


# ##### Birch

# In[35]:


visualization.plot_boxplot(algorithms[1].instances[0])


# ##### DBSCAN

# In[36]:


visualization.plot_boxplot(algorithms[2].instances[0])


# ##### Mean Shift

# In[37]:


visualization.plot_boxplot(algorithms[3].instances[0])


# ##### Ward

# In[38]:


visualization.plot_boxplot(algorithms[4].instances[0])


# #### Dendrograma

# ##### Ward

# In[39]:


visualization.plot_dendrogram(algorithms[4].instances[0])


# In[76]:


from scipy.cluster import hierarchy
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from collections import Counter as counter

def plot_dendrogram_2(instance, p=None, show=True, save_route=None):
    if not instance['ward_specific']:
        raise ValueError('Must have executed ward_specific first')
        
    plt.figure(1)
    plt.clf()
    hierarchy.dendrogram(instance['ward_specific']['linkage_array'], orientation="left", p=p, truncate_mode='lastp' if p else None)
    if save_route: plt.savefig(save_route)
    if show: plt.show()

    plt.clf()


# In[77]:


visualization.plot_dendrogram(algorithms[4].instances[0], p=10)


# In[40]:


visualization.plot_dendrogram_heat(algorithms[4].instances[0])


# ### Análisis K-Means

# In[69]:


kmeans_ch = pd.DataFrame(columns=case.columns)
kmeans_X_clusters = algorithms[0].instances[0]['X_clusters']
for cluster in algorithms[0].instances[0]['cluster_ids']:
    quantiles = kmeans_X_clusters[kmeans_X_clusters['cluster']==cluster].quantile([.25, .75])
    d = {}
    for col in case.columns:
        d[col] = f"{quantiles.loc[.25][col]:.2f}-{quantiles.loc[.75][col]:.2f}"
    kmeans_ch.loc[cluster] = d
print(kmeans_ch.to_latex())


# In[41]:


kmeans = ClusterAlgorithm(KMeans, name='K-Means', centroid_attr='cluster_centers_', not_instantiate=True)


# In[42]:


kmeans.add_instances_product({
    'init': ['k-means++'],
    'n_clusters': range(2, 20),
    'n_init': [5],
    'random_state': [common.RANDOM_SEED],
})


# In[43]:


kmeans


# In[44]:


kmeans.run_instances(case, verbose=True)


# In[45]:


kmeans.calculate_metrics(cluster.metrics, verbose=True)


# In[46]:


kmeans_metrics = pd.DataFrame(columns=cluster.metrics.keys())
for i, instance in enumerate(kmeans.instances):
    kmeans_metrics.loc[i] = instance['metrics']
kmeans_metrics


# In[71]:


print(kmeans_metrics[['Número de clusters', 'Tiempo', 'Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette']].to_latex(index=False))


# In[47]:


sns.lineplot(data=kmeans_metrics, x="Número de clusters", y="Calinski-Harabasz")


# In[48]:


sns.lineplot(data=kmeans_metrics, x="Número de clusters", y="Davies-Bouldin")


# In[49]:


sns.lineplot(data=kmeans_metrics, x="Número de clusters", y="Silhouette")


# ### Análisis Ward

# In[79]:


ward_ch = pd.DataFrame(columns=case.columns)
ward_X_clusters = algorithms[-1].instances[0]['X_clusters']
for cluster in algorithms[-1].instances[0]['cluster_ids']:
    quantiles = ward_X_clusters[ward_X_clusters['cluster']==cluster].quantile([.25, .75])
    d = {}
    for col in case.columns:
        d[col] = f"{quantiles.loc[.25][col]:.2f}-{quantiles.loc[.75][col]:.2f}"
    ward_ch.loc[cluster] = d
ward_ch

