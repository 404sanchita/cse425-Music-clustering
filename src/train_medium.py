import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from model import HybridConvVAE
from clustering import apply_all_clustering_algorithms, print_clustering_summary
from evaluation import calculate_all_metrics, print_metrics
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs('results/latent_visualization', exist_ok=True)

try:
    checkpoint = torch.load('hybrid_data.pt')
    audio_data = checkpoint['audio']
    text_data = checkpoint['text']
    true_labels = checkpoint['labels']
    print(f"Loaded {len(audio_data)} songs with hybrid features.")
except FileNotFoundError:
    print("Error: 'hybrid_data.pt' not found. Run 'src/hybrid_data.py' first!")
    exit()
    
dataset = TensorDataset(audio_data, text_data, true_labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = HybridConvVAE(latent_dim=32, text_dim=384).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("\nStarting Hybrid VAE Training (Audio + Lyrics Embedding)...")
epochs = 51
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for audio_batch, text_batch, _ in train_loader:
        audio_batch = audio_batch.to(device)
        text_batch = text_batch.to(device)
        
        optimizer.zero_grad()
        recon, mu, logvar = model(audio_batch, text_batch)
        
        recon_loss = F.mse_loss(recon, audio_batch, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(dataset):.2f}")

model.eval()
with torch.no_grad():
    all_audio = audio_data.to(device)
    all_text = text_data.to(device)
    _, latent_features, _ = model(all_audio, all_text)
    latent_np = latent_features.cpu().numpy()

print("\n" + "="*60)
print("Applying Clustering Algorithms")
print("="*60)
clustering_results = apply_all_clustering_algorithms(latent_np, n_clusters=6, standardize=True)
print_clustering_summary(clustering_results)

print("\n" + "="*60)
print("Evaluation Metrics")
print("="*60)

all_metrics = {}
true_labels_np = true_labels.numpy()

for algo_name in ['kmeans', 'agglomerative', 'dbscan']:
    if algo_name in clustering_results:
        algo_labels = clustering_results[algo_name]['labels']
        
        if algo_name == 'dbscan':
            n_clusters = clustering_results[algo_name]['n_clusters']
            n_noise = clustering_results[algo_name]['n_noise']
            if n_clusters < 2 or n_noise > len(true_labels_np) * 0.5:
                print(f"\nSkipping {algo_name} evaluation: {n_clusters} clusters, {n_noise} noise points")
                continue
        
        print(f"\n{algo_name.upper()} Clustering:")
        metrics = calculate_all_metrics(
            clustering_results['data_scaled'],
            algo_labels,
            true_labels_np
        )
        all_metrics[algo_name] = metrics
        print_metrics(metrics, title=f"{algo_name.upper()} Metrics")

print("\nGenerating comparison plots...")
tsne = TSNE(n_components=2, perplexity=min(30, len(latent_np)-1), random_state=42)
viz_data = tsne.fit_transform(latent_np)

languages = ['Bangla', 'English', 'Hindi', 'Korean', 'Japanese', 'Spanish']

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

ax = axes[0, 0]
scatter = ax.scatter(viz_data[:, 0], viz_data[:, 1], c=true_labels_np, cmap='tab10', alpha=0.7, s=50)
ax.set_title("Ground Truth: Organized by Languages", fontsize=14, fontweight='bold')
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
legend = ax.legend(*scatter.legend_elements(), title="Languages", loc='best')
for i, name in enumerate(languages):
    if i < len(legend.get_texts()):
        legend.get_texts()[i].set_text(name)

if 'kmeans' in clustering_results:
    ax = axes[0, 1]
    kmeans_labels = clustering_results['kmeans']['labels']
    scatter = ax.scatter(viz_data[:, 0], viz_data[:, 1], c=kmeans_labels, cmap='prism', alpha=0.7, s=50)
    ax.set_title("K-Means Clustering", fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    if 'kmeans' in all_metrics:
        metrics = all_metrics['kmeans']
        subtitle = f"Silhouette: {metrics['silhouette_score']:.3f} | ARI: {metrics['adjusted_rand_index']:.3f}"
        ax.text(0.5, -0.1, subtitle, transform=ax.transAxes, ha='center', fontsize=10)

if 'agglomerative' in clustering_results:
    ax = axes[1, 0]
    agg_labels = clustering_results['agglomerative']['labels']
    scatter = ax.scatter(viz_data[:, 0], viz_data[:, 1], c=agg_labels, cmap='viridis', alpha=0.7, s=50)
    ax.set_title("Agglomerative Clustering", fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    if 'agglomerative' in all_metrics:
        metrics = all_metrics['agglomerative']
        subtitle = f"Silhouette: {metrics['silhouette_score']:.3f} | ARI: {metrics['adjusted_rand_index']:.3f}"
        ax.text(0.5, -0.1, subtitle, transform=ax.transAxes, ha='center', fontsize=10)

if 'dbscan' in clustering_results:
    dbscan_labels = clustering_results['dbscan']['labels']
    n_clusters = clustering_results['dbscan']['n_clusters']
    if n_clusters >= 2:
        ax = axes[1, 1]
        scatter = ax.scatter(viz_data[:, 0], viz_data[:, 1], c=dbscan_labels, cmap='Set3', alpha=0.7, s=50)
        ax.set_title(f"DBSCAN Clustering ({n_clusters} clusters)", fontsize=14, fontweight='bold')
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        if 'dbscan' in all_metrics:
            metrics = all_metrics['dbscan']
            subtitle = f"Silhouette: {metrics['silhouette_score']:.3f} | ARI: {metrics['adjusted_rand_index']:.3f}"
            ax.text(0.5, -0.1, subtitle, transform=ax.transAxes, ha='center', fontsize=10)
    else:
        axes[1, 1].text(0.5, 0.5, 'DBSCAN: Too many noise points\nor insufficient clusters', 
                       transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=12)
        axes[1, 1].set_title("DBSCAN Clustering", fontsize=14, fontweight='bold')
else:
    axes[1, 1].text(0.5, 0.5, 'DBSCAN results not available', 
                   transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=12)
    axes[1, 1].set_title("DBSCAN Clustering", fontsize=14, fontweight='bold')

plt.suptitle("Medium Task: Hybrid ConvVAE - Clustering Comparison", fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/latent_visualization/medium_clustering_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved as 'results/latent_visualization/medium_clustering_comparison.png'")
plt.close()

metrics_df_data = []
for algo_name, metrics in all_metrics.items():
    row = {'algorithm': algo_name}
    row.update(metrics)
    metrics_df_data.append(row)

if metrics_df_data:
    metrics_df = pd.DataFrame(metrics_df_data)
    
    column_order = ['algorithm', 'silhouette_score', 'calinski_harabasz_index', 
                    'davies_bouldin_index', 'adjusted_rand_index', 'nmi', 'cluster_purity']
    existing_columns = [col for col in column_order if col in metrics_df.columns]
    other_columns = [col for col in metrics_df.columns if col not in column_order]
    metrics_df = metrics_df[existing_columns + other_columns]
    
    csv_path = 'results/clustering_metrics.csv'
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        metrics_df['task'] = 'medium'
        existing_df['task'] = existing_df.get('task', 'unknown')
        combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
    else:
        metrics_df['task'] = 'medium'
        metrics_df.to_csv(csv_path, index=False)
    
    print(f"\nMetrics saved to {csv_path}")

print("\n" + "="*60)
print("Medium Task Complete!")
print("="*60)
