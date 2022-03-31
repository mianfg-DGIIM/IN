#!/usr/bin/env python
# coding: utf-8

# # `IN` Práctica 2. Análisis relacional mediante segmentación

# Miguel Ángel Fernández Gutiérrez <<mianfg@correo.ugr.es>>

# ## Caso 1. Ayudas a hogares andaluces

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


# **Filtrado:** hogares andaluces

# In[3]:


case = data[data['DB040']=='ES61']


# **Selección de variables** y renombrado

# In[6]:


case_columns = {
    'HY020': 'renta_disponible',
    'HY050N': 'ayuda_familia_hijos',
    'HY060N': 'asistencia_social',
    'HY070N': 'ayuda_vivienda'
}
case = case.rename(columns=case_columns)
case = case[case_columns.values()]
case


# **Valores perdidos:** no hay valores perdidos

# In[7]:


case.isna().sum()


# **Valores nulos:** hay muchos valores que son 0
# 
# _**Nota:** a continuación se muestran, fijada la variable, el número de instancias que tienen su valor no nulo._

# In[8]:


case.astype(bool).sum(axis=0)


# **Instanciamos y ejecutamos los algoritmos**

# In[9]:


algorithms = [
    ClusterAlgorithm(KMeans, name='K-Means', init='k-means++', n_clusters=5, n_init=5, random_state=common.RANDOM_SEED, centroid_attr="cluster_centers_"),
    ClusterAlgorithm(Birch, name='Birch', branching_factor=25, threshold=0.15, n_clusters=5, centroid_attr="subcluster_centers_"),
    ClusterAlgorithm(DBSCAN, name='DBSCAN', eps=0.15, min_samples=5),
    ClusterAlgorithm(MeanShift, name='MeanShift', centroid_attr="cluster_centers_"),
    ClusterAlgorithm(AgglomerativeClustering, name='Ward', n_clusters=5, linkage='ward')
]


# In[10]:


print("Algoritmos a usar:\n")

for algorithm in algorithms:
    print(algorithm.__repr__())


# In[11]:


print("Ejecutando instancias...\n")

for algorithm in algorithms:
    algorithm.run_instances(case, verbose=True)


# In[12]:


print("Calculando métricas:\n")

for algorithm in algorithms:
    algorithm.calculate_metrics(cluster.metrics, verbose=True)


# In[13]:


case_metrics = pd.DataFrame(columns=cluster.metrics.keys())
for algorithm in algorithms:
    case_metrics.loc[algorithm.algorithm_name] = algorithm.instances[0]['metrics']
case_metrics


# ### Específico de Ward

# In[34]:


cluster.ward_specific(algorithms[4].instances[0])


# ### Visualización

# #### Heatmaps

# ##### K-Means

# In[14]:


visualization.plot_heatmap(algorithms[0].instances[0])


# ##### Birch

# In[15]:


visualization.plot_heatmap(algorithms[1].instances[0])


# ##### DBSCAN

# In[16]:


visualization.plot_heatmap(algorithms[2].instances[0])


# ##### Mean Shift

# In[17]:


visualization.plot_heatmap(algorithms[3].instances[0])


# ##### Ward

# In[18]:


visualization.plot_heatmap(algorithms[4].instances[0])


# #### Cluster sizes

# ##### K-Means

# In[19]:


visualization.plot_cluster_sizes(algorithms[0].instances[0])


# ##### Birch

# In[20]:


visualization.plot_cluster_sizes(algorithms[1].instances[0])


# ##### DBSCAN

# In[21]:


visualization.plot_cluster_sizes(algorithms[2].instances[0])


# In[162]:


algorithms[2].instances[0]['X_clusters'].groupby(['cluster']).size()


# ##### Mean Shift

# In[22]:


visualization.plot_cluster_sizes(algorithms[3].instances[0])


# ##### Ward

# In[23]:


visualization.plot_cluster_sizes(algorithms[4].instances[0])


# #### Scatter matrix

# ##### K-Means

# In[24]:


visualization.plot_scatter_matrix(algorithms[0].instances[0])


# ##### Birch

# In[25]:


visualization.plot_scatter_matrix(algorithms[1].instances[0])


# ##### DBSCAN

# In[26]:


visualization.plot_scatter_matrix(algorithms[2].instances[0])


# ##### Mean Shift

# In[27]:


visualization.plot_scatter_matrix(algorithms[3].instances[0])


# ##### Ward

# In[28]:


visualization.plot_scatter_matrix(algorithms[4].instances[0])


# #### Boxplot

# ##### K-Means

# In[29]:


visualization.plot_boxplot(algorithms[0].instances[0])


# ##### Birch

# In[30]:


visualization.plot_boxplot(algorithms[1].instances[0])


# ##### DBSCAN

# In[31]:


visualization.plot_boxplot(algorithms[2].instances[0])


# ##### Mean Shift

# In[32]:


visualization.plot_boxplot(algorithms[3].instances[0])


# ##### Ward

# In[33]:


visualization.plot_boxplot(algorithms[4].instances[0])


# #### Dendrograma

# ##### Ward

# In[35]:


visualization.plot_dendrogram(algorithms[4].instances[0])


# In[36]:


visualization.plot_dendrogram_heat(algorithms[4].instances[0])


# ### Análisis K-Means

# In[128]:


kmeans_ch = pd.DataFrame(columns=case.columns)
kmeans_X_clusters = algorithms[0].instances[0]['X_clusters']
for cluster in algorithms[0].instances[0]['cluster_ids']:
    quantiles = kmeans_X_clusters[kmeans_X_clusters['cluster']==cluster].quantile([.25, .75])
    d = {}
    for col in case.columns:
        d[col] = f"{quantiles.loc[.25][col]:.2f}-{quantiles.loc[.75][col]:.2f}"
    kmeans_ch.loc[cluster] = d
print(kmeans_ch.to_latex())


# In[39]:


kmeans = ClusterAlgorithm(KMeans, name='K-Means', centroid_attr='cluster_centers_', not_instantiate=True)


# In[40]:


kmeans.add_instances_product({
    'init': ['k-means++'],
    'n_clusters': range(2, 20),
    'n_init': [5],
    'random_state': [common.RANDOM_SEED],
})


# In[41]:


kmeans


# In[43]:


kmeans.run_instances(case, verbose=True)


# In[46]:


kmeans.calculate_metrics(cluster.metrics, verbose=True)


# In[49]:


kmeans_metrics = pd.DataFrame(columns=cluster.metrics.keys())
for i, instance in enumerate(kmeans.instances):
    kmeans_metrics.loc[i] = instance['metrics']
kmeans_metrics


# In[143]:


print(kmeans_metrics[['Número de clusters', 'Tiempo', 'Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette']].to_latex(index=False))


# In[133]:


sns.lineplot(data=kmeans_metrics, x="Número de clusters", y="Calinski-Harabasz")


# In[54]:


sns.lineplot(data=kmeans_metrics, x="Número de clusters", y="Davies-Bouldin")


# In[55]:


sns.lineplot(data=kmeans_metrics, x="Número de clusters", y="Silhouette")


# ### Análisis Birch

# In[144]:


birch_ch = pd.DataFrame(columns=case.columns)
birch_X_clusters = algorithms[0].instances[0]['X_clusters']
for cluster in algorithms[0].instances[0]['cluster_ids']:
    quantiles = birch_X_clusters[birch_X_clusters['cluster']==cluster].quantile([.25, .75])
    d = {}
    for col in case.columns:
        d[col] = f"{quantiles.loc[.25][col]:.2f}-{quantiles.loc[.75][col]:.2f}"
    birch_ch.loc[cluster] = d
print(birch_ch.to_latex())


# In[88]:


birch = ClusterAlgorithm(Birch, name='Birch', centroid_attr='subcluster_centers_', not_instantiate=True)


# In[89]:


birch.add_instances_product({
    'branching_factor': range(10, 45, 5),
    'threshold': [0.01, 0.05, 0.1, 0.15],
    'n_clusters': [5]
})


# In[90]:


birch


# In[91]:


birch.run_instances(case, verbose=True)


# In[92]:


birch.calculate_metrics(cluster.metrics, verbose=True)


# In[93]:


birch_metrics = pd.DataFrame(columns=['Umbral', 'Factor de ramificación'] + list(cluster.metrics.keys()))
for i, instance in enumerate(birch.instances):
    d = instance['metrics']
    d['Umbral'], d['Factor de ramificación'] = instance['instance_values']['threshold'], instance['instance_values']['branching_factor']
    birch_metrics.loc[i] = d
birch_metrics


# In[146]:


print(birch_metrics.to_latex(index=False))


# In[94]:


# fijado factor de ramificación = 20
sns.lineplot(data=birch_metrics[birch_metrics['Factor de ramificación']==20], x="Umbral", y="Calinski-Harabasz")


# In[95]:


# fijado factor de ramificación = 20
sns.lineplot(data=birch_metrics[birch_metrics['Factor de ramificación']==20], x="Umbral", y="Davies-Bouldin")


# In[96]:


# fijado factor de ramificación = 20
sns.lineplot(data=birch_metrics[birch_metrics['Factor de ramificación']==20], x="Umbral", y="Silhouette")


# In[97]:


# fijado factor de ramificación = 40
sns.lineplot(data=birch_metrics[birch_metrics['Factor de ramificación']==40], x="Umbral", y="Calinski-Harabasz")


# In[98]:


# fijado factor de ramificación = 40
sns.lineplot(data=birch_metrics[birch_metrics['Factor de ramificación']==40], x="Umbral", y="Davies-Bouldin")


# In[99]:


# fijado factor de ramificación = 40
sns.lineplot(data=birch_metrics[birch_metrics['Factor de ramificación']==40], x="Umbral", y="Silhouette")


# In[100]:


# fijado umbral = 0.1
sns.lineplot(data=birch_metrics[birch_metrics['Umbral']==.1], x="Factor de ramificación", y="Calinski-Harabasz")


# In[102]:


# fijado umbral = 0.1
sns.lineplot(data=birch_metrics[birch_metrics['Umbral']==.1], x="Factor de ramificación", y="Davies-Bouldin")


# In[103]:


# fijado umbral = 0.1
sns.lineplot(data=birch_metrics[birch_metrics['Umbral']==.1], x="Factor de ramificación", y="Silhouette")


# In[104]:


# fijado umbral = 0.05
sns.lineplot(data=birch_metrics[birch_metrics['Umbral']==.05], x="Factor de ramificación", y="Calinski-Harabasz")


# In[105]:


# fijado umbral = 0.05
sns.lineplot(data=birch_metrics[birch_metrics['Umbral']==.05], x="Factor de ramificación", y="Davies-Bouldin")


# In[106]:


# fijado umbral = 0.05
sns.lineplot(data=birch_metrics[birch_metrics['Umbral']==.05], x="Factor de ramificación", y="Silhouette")


# ### Análisis adicional: diferencias de ayudas entre Comunidades Autónomas

# In[ ]:




