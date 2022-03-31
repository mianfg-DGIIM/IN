# -*- coding: utf-8 -*-
"""
@author: mianfg
"""

import time
import itertools
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics, preprocessing
from math import floor

from scipy.cluster import hierarchy

from common import RANDOM_SEED, iostyle, normalize_01


def instance_product(instances_parameters_list):
    keys = instances_parameters_list.keys()
    lists = []
    for key in keys:
        lists.append(instances_parameters_list[key])

    print(*lists)

    lists_product = itertools.product(*lists)

    instances = []
    for item in list(lists_product):
        instance = {}
        for key, value in zip(keys, item):
            instance[key] = value
        instances.append(instance)

    return instances


class ClusterAlgorithm:
    random_seed = RANDOM_SEED

    def __init__(self, algorithm, name=None, not_instantiate=False, centroid_attr=None,
                 normalizer=normalize_01, **parameters):
        """Class initializer

        Args:
            algorithm (sklearn.cluster): Clustering algorithm, from sklearn.cluster
            name (str, optional): Algorithm name. Defaults to None.
            not_instantiate (bool, optional): Whether to instantiate on init or not.
                Defaults to False.
            centroid_attr (str, optional): Name of sklearn.cluster's attribute to retrieve
                centroids. Defaults to None.
            normalizer (function): Function to normalize input. Must be able to pass as argument
                to pd.DataFrame.apply
            parameters: Parameters for first algorithm instantiation
                (not valid if not_instantiate=True)
        """
        self.algorithm_name = name if name else algorithm.__name__
        self.algorithm = algorithm
        self.centroid_attr = centroid_attr
        self.instances = []
        self.normalizer = normalizer
        if not not_instantiate:
            self.add_instance(**parameters)

    def _get_instance_str(self, i):
        """Generate instance string for documentation (internal)

        Args:
            i (int): Instance index

        Returns:
            str: Index representation as string
        """
        if len(self.instances[i]['instance_values']) == 0:
            return '(no parameters)'
        else:
            return ", ".join([f'{key}={val}' for key, val
                              in self.instances[i]['instance_values'].items()])

    def __str__(self):
        return f"<ClusterAlgorithm [{self.algorithm_name}], " \
            + f"{len(self.instances)} instance{'s' if len(self.instances) != 1 else ''}>"

    def __repr__(self):
        rep = f"<ClusterAlgorithm [{self.algorithm_name}], " \
            + f"{len(self.instances)} instance{'s' if len(self.instances) != 1 else ''}:\n"
        for i in range(len(self.instances)):
            rep += f"\t{self._get_instance_str(i)};\n"
        rep = rep[:-2]
        rep += '>'
        return rep

    def add_instance(self, **parameters):
        """Add algorithm instance

        Args:
            parameters: Parameters to pass to the algorithm's constructor
        """
        self.instances.append({
            'algorithm_instance': self.algorithm(**parameters),
            'instance_values': parameters
        })

    def add_instances(self, instances_parameters):
        """Add multiple algorithm instances

        Args:
            instances_parameters (list<dict>): List of parameters for each algorithm's
                instantiation, passed as list of dict
        """
        for parameters in instances_parameters:
            self.add_instance(**parameters)

    def add_instances_product(self, instances_parameters_list):
        """Add multiple algorithm instances as cartesian product

        Parameters are passed via instance_parameters_list. For example, having:
            instances_parameters_list = {'a': [1, 3], 'b': range(1, 3)}
        Will create the following instances:
            - a=1, b=1
            - a=1, b=2
            - a=3, b=1
            - a=3, b=2

        Args:
            instances_parameters_list (dict): Parameters for algorithm's instantiation
                via cartesian product, passed as dict with str keys and list values
        """
        self.add_instances(instance_product(instances_parameters_list))

    def only_one_instance(self):
        """Whether the algorithm has only one instance

        Returns:
            bool: True if the algorithm has only one instance
        """
        return len(self.instances) == 1

    def get_instances(self):
        """Get all algorithm instances

        Returns:
            list<sklearn.cluster>: List of algorithm instances
        """
        return [instance['algorithm_instance'] for instance in self.instances]
    
    def unnorm_centroids(self, instance):
        centroids_norm = getattr(instance['algorithm_instance'], self.centroid_attr)
        centroids_pd = pd.DataFrame(centroids_norm.copy(), columns=list(instance['X']))
        centroids_unnorm = centroids_pd.copy()
        for c in list(centroids_pd):
            centroids_unnorm[c] = instance['X'][c].min() + centroids_pd[c] * (instance['X'][c].max() - instance['X'][c].min())
        return centroids_unnorm

    def calculate_centroids(self, instance):
        centroids = pd.DataFrame(columns=instance['X'].columns)
        for cluster_id in instance['cluster_ids']:
            filtered = instance['X_clusters'][instance['X_clusters']['cluster']==cluster_id]
            centroids.loc[len(centroids)] = list(np.array(filtered.sum()[:-1])/filtered.shape[0])
        return centroids

    def run_instances(self, X, verbose=False):
        """Run all algorithm instances

        Args:
            X (pd.DataFrame): [description]
            verbose (bool, optional): [description]. Defaults to False.
        """
        # X should be normalized
        for i, instance in enumerate(self.instances):
            if verbose:
                print(f"[{self.algorithm_name}] Running instance "
                      + f"{self._get_instance_str(i)}...")
            t = time.time()
            instance['X'] = X.copy()
            instance['X_norm'] = X.copy().apply(self.normalizer) if self.normalizer else X.copy()
            instance['results'] = instance['algorithm_instance'].fit_predict(
                instance['X_norm'])
            instance['time'] = time.time() - t
            instance['clusters'] = pd.DataFrame(
                instance['results'], index=instance['X'].index, columns=['cluster'])
            instance['cluster_ids'] = list(np.unique(instance['clusters']['cluster']))
            instance['num_clusters'] = len(instance['cluster_ids'])
            instance['X_clusters'] = pd.concat(
                [instance['X'], instance['clusters']], axis=1)
            if self.centroid_attr:
                instance['centroids'] = self.unnorm_centroids(instance)
            else:
                instance['centroids'] = self.calculate_centroids(instance)
        if verbose:
            print(f"{iostyle.OKBLUE}[{self.algorithm_name}] "
                  + f"Instances run successfully{iostyle.ENDC}")

    def calculate_metrics(self, metrics, verbose=False):
        for i, instance in enumerate(self.instances):
            instance['metrics'] = {}
            if verbose:
                print(f"[{self.algorithm_name}] "
                      + f"Calculating metrics in instance {self._get_instance_str(i)}...")
            error = False
            for metric_name in metrics:
                try:
                    instance['metrics'][metric_name] = metrics[metric_name](instance)
                except Exception as e:
                    print(
                        f"{iostyle.WARNING}[ERROR] Could not calculate Silhouette score")
                    print("       ", f"({type(e).__name__})", str(e), iostyle.ENDC)
                    instance['metrics'][metric_name] = None
                    error = True
        if verbose:
            print(f"{iostyle.FAIL if error else iostyle.OKBLUE}[{self.algorithm_name}] "
                  + "Metrics calculated " + ("with errors" if error else "successfully") + iostyle.ENDC)


# ====================
# MÉTRICAS
# ====================

# Tiempo
def metric_time(instance):
    return instance['time']

# Calinski-Harabasz
def metric_ch(instance):
    return sk_metrics.calinski_harabasz_score(instance['X_norm'], instance['results'])

# Davies-Bouldin
def metric_db(instance):
    return sk_metrics.davies_bouldin_score(instance['X_norm'], instance['results'])

# Silhouette
def metric_silhouette(instance):
    sample = 0.2 if len(instance['X_norm']) > 10**5 else 1
    return sk_metrics.silhouette_score(instance['X_norm'], instance['results'], metric='euclidean',
                                       sample_size=floor(sample*len(instance['X_norm'])), random_state=RANDOM_SEED)

# número de clusters
def metric_num_clusters(instance):
    return instance['num_clusters']


metrics = {
    'Tiempo': metric_time,
    'Calinski-Harabasz': metric_ch,
    'Davies-Bouldin': metric_db,
    'Silhouette': metric_silhouette,
    'Número de clusters': metric_num_clusters
}


# ====================
# ESPECÍFICOS
# ====================

def ward_specific(instance, min_size=10, verbose=True):
    """Additional processing for Ward

    Args:
        instance (ClusterAlgorithm.instance): Algorithm instance.
            Must be of sklearn.cluster.AgglomerativeClustering (Ward)
    """
    labels = instance['algorithm_instance'].labels_
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    if min_size > 0:
        X_filtered = instance['X_clusters'][instance['X_clusters'].groupby('cluster').cluster.transform(len) > min_size]
        num_clusters_filtered = len(set(X_filtered['cluster']))
        if verbose: print(f"Ward filtering applied (min_size={min_size}):\n" \
            + f" - Number of clusters before filtering: {num_clusters}\n" \
            + f" - Number of clusters after filtering:  {num_clusters_filtered}\n" \
            + f" - Size of dataset before filtering:    {len(instance['X_clusters'])}\n" \
            + f" - Size of dataset after filtering:     {len(X_filtered)}\n")
        X_filtered = X_filtered.drop('cluster', 1)
    else:
        X_filtered = instance['X_clusters']
        num_clusters_filtered = num_clusters
        if verbose: print(f"No Ward filtering applied:\n" \
            + f" - Number of clusters: {num_clusters}\n" \
            + f" - Size of dataset: {  len(X_filtered)}\n")
    
    X_filtered_norm = preprocessing.normalize(X_filtered, norm='l2')
    linkage_array = hierarchy.ward(X_filtered_norm)

    instance['ward_specific'] = {
        'X_filtered':      X_filtered,
        'X_filtered_norm': X_filtered_norm,
        'linkage_array':   linkage_array
    }
