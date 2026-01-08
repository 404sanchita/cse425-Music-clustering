
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def calculate_silhouette_score(data, labels):
   
    return silhouette_score(data, labels)


def calculate_calinski_harabasz_index(data, labels):

    return calinski_harabasz_score(data, labels)


def calculate_davies_bouldin_index(data, labels):
 
    return davies_bouldin_score(data, labels)


def calculate_adjusted_rand_index(true_labels, pred_labels):
  
    return adjusted_rand_score(true_labels, pred_labels)


def calculate_nmi(true_labels, pred_labels):

    return normalized_mutual_info_score(true_labels, pred_labels)


def calculate_cluster_purity(true_labels, pred_labels):

    n_samples = len(true_labels)
    
    
    clusters = np.unique(pred_labels)
    classes = np.unique(true_labels)
    
    
    correct = 0
    for cluster_id in clusters:
        
        cluster_mask = pred_labels == cluster_id
        if not np.any(cluster_mask):
            continue
            
        
        cluster_true_labels = true_labels[cluster_mask]
        class_counts = {}
        for class_id in classes:
            class_counts[class_id] = np.sum(cluster_true_labels == class_id)
        
    
        correct += max(class_counts.values())
    
    purity = correct / n_samples
    return purity


def calculate_all_metrics(data, pred_labels, true_labels=None):
  
    metrics = {}
    

    metrics['silhouette_score'] = calculate_silhouette_score(data, pred_labels)
    metrics['calinski_harabasz_index'] = calculate_calinski_harabasz_index(data, pred_labels)
    metrics['davies_bouldin_index'] = calculate_davies_bouldin_index(data, pred_labels)
    
    if true_labels is not None:
        metrics['adjusted_rand_index'] = calculate_adjusted_rand_index(true_labels, pred_labels)
        metrics['nmi'] = calculate_nmi(true_labels, pred_labels)
        metrics['cluster_purity'] = calculate_cluster_purity(true_labels, pred_labels)
    
    return metrics


def print_metrics(metrics, title="Clustering Metrics"):
    
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:30s}: {value:.4f}")
    
    print(f"{'='*50}\n")

