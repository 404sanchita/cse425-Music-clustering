import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler


def kmeans_clustering(data, n_clusters=6, random_state=42, n_init=10):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(data)
    return labels, kmeans


def agglomerative_clustering(data, n_clusters=6, linkage='ward', metric='euclidean'):

    if linkage == 'ward':
        agg_clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
    else:

        try:
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=metric
            )
        except TypeError:
        
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                affinity=metric
            )
    
    labels = agg_clustering.fit_predict(data)
    return labels, agg_clustering


def dbscan_clustering(data, eps=0.5, min_samples=5, metric='euclidean'):
   
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(data)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    return labels, dbscan, n_clusters, n_noise


def apply_all_clustering_algorithms(data, n_clusters=6, standardize=True):
    
    results = {}

    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data
        scaler = None
    
    kmeans_labels, kmeans_model = kmeans_clustering(data_scaled, n_clusters=n_clusters)
    results['kmeans'] = {
        'labels': kmeans_labels,
        'model': kmeans_model,
        'n_clusters': n_clusters
    }
    
    agg_labels, agg_model = agglomerative_clustering(data_scaled, n_clusters=n_clusters)
    results['agglomerative'] = {
        'labels': agg_labels,
        'model': agg_model,
        'n_clusters': n_clusters
    }
    
    from sklearn.neighbors import NearestNeighbors
    min_samples_dbscan = 5
    neighbors = NearestNeighbors(n_neighbors=min_samples_dbscan)
    neighbors_fit = neighbors.fit(data_scaled)
    distances, indices = neighbors_fit.kneighbors(data_scaled)
    distances = np.sort(distances, axis=0)
    distances = distances[:, min_samples_dbscan-1]
    eps_heuristic = np.median(distances) * 0.5
    
    dbscan_labels, dbscan_model, n_dbscan_clusters, n_noise = dbscan_clustering(
        data_scaled, 
        eps=eps_heuristic, 
        min_samples=min_samples_dbscan
    )
    results['dbscan'] = {
        'labels': dbscan_labels,
        'model': dbscan_model,
        'n_clusters': n_dbscan_clusters,
        'n_noise': n_noise,
        'eps_used': eps_heuristic
    }
    
    results['scaler'] = scaler
    results['data_scaled'] = data_scaled
    
    return results


def print_clustering_summary(results):
    
    print("\n" + "="*60)
    print("Clustering Results Summary")
    print("="*60)
    
    for algo_name in ['kmeans', 'agglomerative', 'dbscan']:
        if algo_name in results:
            algo_results = results[algo_name]
            print(f"\n{algo_name.upper()}:")
            print(f"  Number of clusters: {algo_results['n_clusters']}")
            if algo_name == 'dbscan':
                print(f"  Number of noise points: {algo_results['n_noise']}")
                print(f"  EPS used: {algo_results['eps_used']:.4f}")
            labels = algo_results['labels']
            unique_labels = np.unique(labels)
            print(f"  Cluster labels: {unique_labels}")
            for label in unique_labels:
                count = np.sum(labels == label)
                print(f"    Cluster {label}: {count} samples")
    
    print("="*60 + "\n")

