import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from model import HardMusicCVAE
from clustering import kmeans_clustering
from evaluation import calculate_all_metrics, print_metrics
import pandas as pd

class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=32, text_dim=384):
        super(SimpleAutoencoder, self).__init__()
        
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        combined_dim = 16384 + text_dim
        self.fc_latent = nn.Linear(combined_dim, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, audio, text):
        a_feat = self.audio_encoder(audio)
        combined = torch.cat([a_feat, text], dim=1)
        z = self.fc_latent(combined)
        recon = self.decoder(self.decoder_input(z))
        return recon, z


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs('results/latent_visualization', exist_ok=True)

checkpoint = torch.load('hybrid_data.pt')
audio_data = checkpoint['audio']
text_data = checkpoint['text']
true_labels = checkpoint['labels']
labels = checkpoint['labels']

print(f"Loaded {len(audio_data)} songs with hybrid features.")

one_hot_labels = F.one_hot(labels, num_classes=6).float()

dataset = TensorDataset(audio_data, text_data, one_hot_labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = HardMusicCVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("\n" + "="*60)
print("Training Conditional VAE (Hard Task)")
print("="*60)

for epoch in range(61):
    model.train()
    total_loss = 0
    for audio, text, oh_labels in train_loader:
        audio, text, oh_labels = audio.to(device), text.to(device), oh_labels.to(device)
        
        optimizer.zero_grad()
        recon, mu, logvar = model(audio, text, oh_labels)
        
        recon_loss = F.mse_loss(recon, audio, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(dataset):.2f}")

torch.save(model.state_dict(), 'hard_model.pth')
print("Model saved to 'hard_model.pth'")

model.eval()
with torch.no_grad():
    all_audio = audio_data.to(device)
    all_text = text_data.to(device)
    all_oh_labels = one_hot_labels.to(device)
    
    recon_cvae, latent_cvae, _ = model(all_audio, all_text, all_oh_labels)
    latent_cvae_np = latent_cvae.cpu().numpy()

print(f"\nExtracted latent representations: {latent_cvae_np.shape}")

print("\n" + "="*60)
print("Training Autoencoder Baseline")
print("="*60)

ae_model = SimpleAutoencoder(latent_dim=32, text_dim=384).to(device)
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-4)

for epoch in range(51):
    ae_model.train()
    total_loss = 0
    for audio, text, _ in train_loader:
        audio, text = audio.to(device), text.to(device)
        
        ae_optimizer.zero_grad()
        recon, latent = ae_model(audio, text)
        loss = F.mse_loss(recon, audio, reduction='sum')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae_model.parameters(), 1.0)
        ae_optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(dataset):.2f}")

ae_model.eval()
with torch.no_grad():
    _, latent_ae = ae_model(all_audio, all_text)  
    latent_ae_np = latent_ae.cpu().numpy()
    
print(f"Autoencoder latent shape: {latent_ae_np.shape}")

spectral_features = audio_data.squeeze(1).reshape(len(audio_data), -1).numpy()
print(f"\nSpectral features shape: {spectral_features.shape}")

pca_spectral = PCA(n_components=32, random_state=42)
spectral_pca = pca_spectral.fit_transform(spectral_features)

print("\n" + "="*60)
print("Clustering Evaluation")
print("="*60)

true_labels_np = true_labels.numpy()
n_clusters = 6

all_results = {}

print("\n1. CVAE + K-Means:")
cvae_labels, _ = kmeans_clustering(latent_cvae_np, n_clusters=n_clusters, random_state=42)
all_results['CVAE + K-Means'] = {
    'latent': latent_cvae_np,
    'labels': cvae_labels
}

print("\n2. Autoencoder + K-Means:")
ae_labels, _ = kmeans_clustering(latent_ae_np, n_clusters=n_clusters, random_state=42)
all_results['Autoencoder + K-Means'] = {
    'latent': latent_ae_np,
    'labels': ae_labels
}

print("\n3. PCA + K-Means (Spectral Features):")
pca_labels, _ = kmeans_clustering(spectral_pca, n_clusters=n_clusters, random_state=42)
all_results['PCA + K-Means'] = {
    'latent': spectral_pca,
    'labels': pca_labels
}

print("\n4. Direct Spectral Feature + K-Means:")
if spectral_features.shape[1] > 1000:
    spectral_direct_labels = pca_labels
    spectral_direct_latent = spectral_pca
else:
    spectral_direct_labels, _ = kmeans_clustering(spectral_features, n_clusters=n_clusters, random_state=42)
    spectral_direct_latent = spectral_features

all_results['Direct Spectral + K-Means'] = {
    'latent': spectral_direct_latent,
    'labels': spectral_direct_labels
}

print("\n" + "="*60)
print("Evaluation Metrics")
print("="*60)

all_metrics = {}
for method_name, results in all_results.items():
    print(f"\n{method_name}:")
    metrics = calculate_all_metrics(
        results['latent'],
        results['labels'],
        true_labels_np
    )
    all_metrics[method_name] = metrics
    print_metrics(metrics, title=f"{method_name} Metrics")

print("\nGenerating t-SNE visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

methods_list = [
    ('CVAE + K-Means', all_results['CVAE + K-Means']),
    ('Autoencoder + K-Means', all_results['Autoencoder + K-Means']),
    ('PCA + K-Means', all_results['PCA + K-Means']),
    ('Direct Spectral + K-Means', all_results['Direct Spectral + K-Means'])
]

for idx, (method_name, results) in enumerate(methods_list):
    ax = axes[idx // 2, idx % 2]
    
    tsne = TSNE(n_components=2, perplexity=min(30, len(results['latent'])-1), random_state=42)
    viz_data = tsne.fit_transform(results['latent'])
    
    scatter = ax.scatter(viz_data[:, 0], viz_data[:, 1], c=results['labels'], 
                        cmap='tab10', alpha=0.7, s=50)
    ax.set_title(method_name, fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    
    if method_name in all_metrics:
        metrics = all_metrics[method_name]
        subtitle = f"NMI: {metrics['nmi']:.3f} | Purity: {metrics['cluster_purity']:.3f}"
        ax.text(0.5, -0.1, subtitle, transform=ax.transAxes, ha='center', fontsize=10)

plt.suptitle("Hard Task: Latent Space Comparison (t-SNE)", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('results/latent_visualization/hard_latent_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/latent_visualization/hard_latent_comparison.png")
plt.close()

print("\nGenerating cluster distribution plots...")
languages = ['Bangla', 'English', 'Hindi', 'Korean', 'Japanese', 'Spanish']

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

for idx, (method_name, results) in enumerate(methods_list):
    ax = axes[idx // 2, idx % 2]
    
    labels_pred = results['labels']
    
    cluster_dist = np.zeros((n_clusters, len(languages)))
    for cluster_id in range(n_clusters):
        cluster_mask = labels_pred == cluster_id
        cluster_true_labels = true_labels_np[cluster_mask]
        for lang_id in range(len(languages)):
            cluster_dist[cluster_id, lang_id] = np.sum(cluster_true_labels == lang_id)

    cluster_dist_pct = cluster_dist / (cluster_dist.sum(axis=1, keepdims=True) + 1e-6) * 100
    
    im = ax.imshow(cluster_dist_pct, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(languages, rotation=45, ha='right')
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)])
    ax.set_title(f"{method_name}\nCluster Distribution over Languages", 
                fontsize=12, fontweight='bold')
    ax.set_xlabel("Language")
    ax.set_ylabel("Cluster")
    
    for i in range(n_clusters):
        for j in range(len(languages)):
            text = ax.text(j, i, f'{cluster_dist_pct[i, j]:.1f}%',
                         ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Percentage (%)')

plt.suptitle("Hard Task: Cluster Distribution over Languages", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('results/latent_visualization/hard_cluster_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: results/latent_visualization/hard_cluster_distribution.png")
plt.close()

print("\nGenerating reconstruction examples...")
n_examples_per_lang = 2
examples = []

for lang_id in range(len(languages)):
    lang_mask = true_labels_np == lang_id
    lang_indices = np.where(lang_mask)[0]
    if len(lang_indices) >= n_examples_per_lang:
        selected = np.random.choice(lang_indices, n_examples_per_lang, replace=False)
        examples.extend(selected)

examples = examples[:12]

fig, axes = plt.subplots(3, 8, figsize=(24, 9))

model.eval()
with torch.no_grad():
    for idx, example_idx in enumerate(examples):
        if idx >= 12:
            break
        
        audio_ex = audio_data[example_idx:example_idx+1].to(device)
        text_ex = text_data[example_idx:example_idx+1].to(device)
        oh_label_ex = one_hot_labels[example_idx:example_idx+1].to(device)
        
        recon, _, _ = model(audio_ex, text_ex, oh_label_ex)
        
        original = audio_ex[0, 0].cpu().numpy()
        reconstructed = recon[0, 0].cpu().numpy()
        
        ax = axes[idx // 4, (idx % 4) * 2]
        ax.imshow(original, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Original\n(Lang: {languages[true_labels_np[example_idx]]})", fontsize=9)
        ax.axis('off')
        
        ax = axes[idx // 4, (idx % 4) * 2 + 1]
        ax.imshow(reconstructed, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Reconstructed\n(Cluster: {cvae_labels[example_idx]})", fontsize=9)
        ax.axis('off')

plt.suptitle("Hard Task: Reconstruction Examples", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/latent_visualization/hard_reconstructions.png', dpi=300, bbox_inches='tight')
print("Saved: results/latent_visualization/hard_reconstructions.png")
plt.close()

print("\nSaving metrics to CSV...")
metrics_df_data = []
for method_name, metrics in all_metrics.items():
    row = {'method': method_name, 'task': 'hard'}
    row.update(metrics)
    metrics_df_data.append(row)

if metrics_df_data:
    metrics_df = pd.DataFrame(metrics_df_data)
    
    column_order = ['method', 'task', 'silhouette_score', 'calinski_harabasz_index',
                    'davies_bouldin_index', 'adjusted_rand_index', 'nmi', 'cluster_purity']
    existing_columns = [col for col in column_order if col in metrics_df.columns]
    other_columns = [col for col in metrics_df.columns if col not in column_order]
    metrics_df = metrics_df[existing_columns + other_columns]
    
    csv_path = 'results/clustering_metrics.csv'
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
    else:
        metrics_df.to_csv(csv_path, index=False)
    
    print(f"Metrics saved to {csv_path}")

print("\n" + "="*60)
print("Hard Task Complete!")
print("="*60)
print("\nGenerated files:")
print("  - results/latent_visualization/hard_latent_comparison.png")
print("  - results/latent_visualization/hard_cluster_distribution.png")
print("  - results/latent_visualization/hard_reconstructions.png")
print("  - results/clustering_metrics.csv")
