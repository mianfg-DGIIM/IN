#!/usr/bin/env python
# coding: utf-8

# # `IN` Práctica 2. Análisis relacional mediante segmentación

# Miguel Ángel Fernández Gutiérrez <<mianfg@correo.ugr.es>>

# ## Caso 3. Alquiler de segundas viviendas y poder adquisitivo

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


# **Filtrado:** dispone de segunda vivienda

# In[3]:


case = data[data['HV020']==1]


# **Selección de variables** y renombrado

# In[4]:


case_columns = {
    'HY020': 'renta_disponible',
    'HY040N': 'renta_alquiler',
    'HV010': 'valor_vivienda',
    'HS130': 'ingresos_minimos'
}
case = case.rename(columns=case_columns)
case = case[case_columns.values()]
case


# **Valores perdidos:** hay valores perdidos. En este caso lo que vamos a hacer no es imputar la media, sino eliminarlos

# In[5]:


case.isna().sum()


# In[7]:


case = case[case.valor_vivienda.notnull() & case.ingresos_minimos.notnull()]
case


# In[8]:


case.isna().sum()


# **Valores nulos:** hay muchos valores que son 0 en `renta_alquiler`
# 
# _**Nota:** a continuación se muestran, fijada la variable, el número de instancias que tienen su valor no nulo._

# In[9]:


case.astype(bool).sum(axis=0)


# **Instanciamos y ejecutamos los algoritmos**

# In[134]:


# nota: en Birch hemos tenido que decrementar el threshold
algorithms = [
    ClusterAlgorithm(KMeans, name='K-Means', init='k-means++', n_clusters=5, n_init=5, random_state=common.RANDOM_SEED, centroid_attr="cluster_centers_"),
    ClusterAlgorithm(Birch, name='Birch', branching_factor=25, threshold=0.1, n_clusters=5, centroid_attr="subcluster_centers_"),
    ClusterAlgorithm(DBSCAN, name='DBSCAN', eps=0.01, min_samples=15),
    ClusterAlgorithm(MeanShift, name='MeanShift', centroid_attr="cluster_centers_"),
    ClusterAlgorithm(AgglomerativeClustering, name='Ward', n_clusters=5, linkage='ward')
]


# In[135]:


print("Algoritmos a usar:\n")

for algorithm in algorithms:
    print(algorithm.__repr__())


# In[136]:


print("Ejecutando instancias...\n")

for algorithm in algorithms:
    algorithm.run_instances(case, verbose=True)


# In[137]:


print("Calculando métricas:\n")

for algorithm in algorithms:
    algorithm.calculate_metrics(cluster.metrics, verbose=True)


# In[138]:


case_metrics = pd.DataFrame(columns=cluster.metrics.keys())
for algorithm in algorithms:
    case_metrics.loc[algorithm.algorithm_name] = algorithm.instances[0]['metrics']
case_metrics


# In[161]:


print(case_metrics.to_latex())


# ### Específico de Ward

# In[139]:


cluster.ward_specific(algorithms[4].instances[0])


# ### Visualización

# #### Heatmaps

# ##### K-Means

# In[140]:


visualization.plot_heatmap(algorithms[0].instances[0])


# ##### Birch

# In[141]:


visualization.plot_heatmap(algorithms[1].instances[0])


# ##### DBSCAN

# In[142]:


visualization.plot_heatmap(algorithms[2].instances[0])


# ##### Mean Shift

# In[143]:


visualization.plot_heatmap(algorithms[3].instances[0])


# ##### Ward

# In[144]:


visualization.plot_heatmap(algorithms[4].instances[0])


# #### Cluster sizes

# ##### K-Means

# In[145]:


visualization.plot_cluster_sizes(algorithms[0].instances[0])


# ##### Birch

# In[146]:


visualization.plot_cluster_sizes(algorithms[1].instances[0])


# ##### DBSCAN

# In[147]:


visualization.plot_cluster_sizes(algorithms[2].instances[0])


# ##### Mean Shift

# In[148]:


visualization.plot_cluster_sizes(algorithms[3].instances[0])


# ##### Ward

# In[149]:


visualization.plot_cluster_sizes(algorithms[4].instances[0])


# #### Scatter matrix

# ##### K-Means

# In[150]:


visualization.plot_scatter_matrix(algorithms[0].instances[0])


# ##### Birch

# In[151]:


visualization.plot_scatter_matrix(algorithms[1].instances[0])


# ##### DBSCAN

# In[152]:


visualization.plot_scatter_matrix(algorithms[2].instances[0])


# ##### Mean Shift

# In[153]:


visualization.plot_scatter_matrix(algorithms[3].instances[0])


# ##### Ward

# In[154]:


visualization.plot_scatter_matrix(algorithms[4].instances[0])


# #### Boxplot

# ##### K-Means

# In[155]:


visualization.plot_boxplot(algorithms[0].instances[0])


# ##### Birch

# In[156]:


visualization.plot_boxplot(algorithms[1].instances[0])


# ##### DBSCAN

# In[157]:


visualization.plot_boxplot(algorithms[2].instances[0])


# ##### Mean Shift

# In[158]:


visualization.plot_boxplot(algorithms[3].instances[0])


# ##### Ward

# In[159]:


visualization.plot_boxplot(algorithms[4].instances[0])


# #### Dendrograma

# ##### Ward

# In[45]:


visualization.plot_dendrogram(algorithms[4].instances[0])


# In[46]:


visualization.plot_dendrogram_heat(algorithms[4].instances[0])


# ### Análisis K-Means

# In[160]:


kmeans_ch = pd.DataFrame(columns=case.columns)
kmeans_X_clusters = algorithms[0].instances[0]['X_clusters']
for cluster in algorithms[0].instances[0]['cluster_ids']:
    quantiles = kmeans_X_clusters[kmeans_X_clusters['cluster']==cluster].quantile([.25, .75])
    d = {}
    for col in case.columns:
        d[col] = f"{quantiles.loc[.25][col]:.2f}-{quantiles.loc[.75][col]:.2f}"
    kmeans_ch.loc[cluster] = d
print(kmeans_ch.to_latex())


# In[47]:


kmeans = ClusterAlgorithm(KMeans, name='K-Means', centroid_attr='cluster_centers_', not_instantiate=True)


# In[48]:


kmeans.add_instances_product({
    'init': ['k-means++'],
    'n_clusters': range(2, 20),
    'n_init': [5],
    'random_state': [common.RANDOM_SEED],
})


# In[49]:


kmeans


# In[50]:


kmeans.run_instances(case, verbose=True)


# In[51]:


kmeans.calculate_metrics(cluster.metrics, verbose=True)


# In[52]:


kmeans_metrics = pd.DataFrame(columns=cluster.metrics.keys())
for i, instance in enumerate(kmeans.instances):
    kmeans_metrics.loc[i] = instance['metrics']
kmeans_metrics


# In[53]:


sns.lineplot(data=kmeans_metrics, x="Número de clusters", y="Calinski-Harabasz")


# In[54]:


sns.lineplot(data=kmeans_metrics, x="Número de clusters", y="Davies-Bouldin")


# In[55]:


sns.lineplot(data=kmeans_metrics, x="Número de clusters", y="Silhouette")


# ### Análisis DBSCAN

# In[164]:


dbscan_ch = pd.DataFrame(columns=case.columns)
dbscan_X_clusters = algorithms[2].instances[0]['X_clusters']
for cluster in algorithms[2].instances[0]['cluster_ids']:
    quantiles = dbscan_X_clusters[dbscan_X_clusters['cluster']==cluster].quantile([.25, .75])
    d = {}
    for col in case.columns:
        d[col] = f"{quantiles.loc[.25][col]:.2f}-{quantiles.loc[.75][col]:.2f}"
    dbscan_ch.loc[cluster] = d
dbscan_ch


# In[97]:


dbscan = ClusterAlgorithm(DBSCAN, name='DBSCAN', not_instantiate=True)


# In[98]:


dbscan.add_instances_product({
    'eps': [.01, .025, .05, .1, .15, .2, .25, .3],
    'min_samples': range(5, 25, 5),
})


# In[99]:


dbscan


# In[100]:


dbscan.run_instances(case, verbose=True)


# In[101]:


dbscan.calculate_metrics(cluster.metrics, verbose=True)


# In[125]:


dbscan_metrics = pd.DataFrame(columns=['Epsilon', 'Mínimo de samples'] + list(cluster.metrics.keys()))
for i, instance in enumerate(dbscan.instances):
    d = instance['metrics']
    d['Epsilon'], d['Mínimo de samples'] = instance['instance_values']['eps'], instance['instance_values']['min_samples']
    dbscan_metrics.loc[i] = d
dbscan_metrics['Número de clusters'] = dbscan_metrics['Número de clusters'].astype(str).astype(int)
dbscan_metrics


# In[126]:


# fijado eps=.01
sns.lineplot(data=dbscan_metrics[dbscan_metrics['Epsilon']==.01], x="Mínimo de samples", y="Calinski-Harabasz")


# In[127]:


# fijado eps=.01
sns.lineplot(data=dbscan_metrics[dbscan_metrics['Epsilon']==.01], x="Mínimo de samples", y="Davies-Bouldin")


# In[128]:


# fijado eps=.01
sns.lineplot(data=dbscan_metrics[dbscan_metrics['Epsilon']==.01], x="Mínimo de samples", y="Silhouette")


# In[129]:


# fijado eps=.01
sns.lineplot(data=dbscan_metrics[dbscan_metrics['Epsilon']==.01], x="Mínimo de samples", y="Número de clusters")


# In[130]:


# fijado min_samples=10
sns.lineplot(data=dbscan_metrics[dbscan_metrics['Mínimo de samples']==10], x="Epsilon", y="Calinski-Harabasz")


# In[131]:


# fijado min_samples=10
sns.lineplot(data=dbscan_metrics[dbscan_metrics['Mínimo de samples']==10], x="Epsilon", y="Davies-Bouldin")


# In[132]:


# fijado min_samples=10
sns.lineplot(data=dbscan_metrics[dbscan_metrics['Mínimo de samples']==10], x="Epsilon", y="Silhouette")


# In[133]:


# fijado min_samples=10
sns.lineplot(data=dbscan_metrics[dbscan_metrics['Mínimo de samples']==10], x="Epsilon", y="Número de clusters")

