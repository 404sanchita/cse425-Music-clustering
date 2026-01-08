import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class MusicVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MusicVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

data = torch.load(r'C:\Users\sanch\MusicClusteringProject\processed_data.pt')
train_loader = DataLoader(data, batch_size=16, shuffle=True)

input_dim = 13  
model = MusicVAE(input_dim=input_dim, hidden_dim=32, latent_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training starting...")
for epoch in range(101):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        recon_loss = F.mse_loss(recon_batch, batch, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (recon_loss + kld_loss)
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch} complete.")

model.eval()
with torch.no_grad():
    _, latent_features, _ = model(data)

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
labels = kmeans.fit_predict(latent_features.numpy())

tsne = TSNE(n_components=2, random_state=42)
viz_data = tsne.fit_transform(latent_features.numpy())
plt.scatter(viz_data[:, 0], viz_data[:, 1], c=labels, cmap='tab10')
plt.title("Music Language Clusters")
plt.savefig('results/latent_visualization/my_clusters.png')
print("Finished! Check 'my_clusters.png'.")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pca = PCA(n_components=2)
pca_features = pca.fit_transform(data.numpy())

kmeans_pca = KMeans(n_clusters=6, n_init=10)
pca_labels = kmeans_pca.fit_predict(pca_features)

vae_s_score = silhouette_score(latent_features.numpy(), labels)
pca_s_score = silhouette_score(pca_features, pca_labels)

print("\n--- FINAL COMPARISON ---")
print(f"VAE Silhouette Score: {vae_s_score:.4f}")
print(f"PCA Silhouette Score: {pca_s_score:.4f}")

vae_ch_score = calinski_harabasz_score(latent_features.numpy(), labels)
pca_ch_score = calinski_harabasz_score(pca_features, pca_labels)

print(f"VAE Calinski-Harabasz Index: {vae_ch_score:.4f}")
print(f"PCA Calinski-Harabasz Index: {pca_ch_score:.4f}")