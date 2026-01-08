import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() 
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
    
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.decoder_input(z)), mu, logvar    
    


class HybridConvVAE(nn.Module):
    def __init__(self, latent_dim=32, text_dim=384):
        super(HybridConvVAE, self).__init__()
        
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
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, audio, text):
        a_feat = self.audio_encoder(audio)
        combined = torch.cat([a_feat, text], dim=1)
        
        mu, logvar = self.fc_mu(combined), self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)
        

        recon = self.decoder(self.decoder_input(z))
        return recon, mu, logvar
    



class ConditionalMusicVAE(nn.Module):
    def __init__(self, latent_dim=32, text_dim=384, num_classes=6):
        super(ConditionalMusicVAE, self).__init__()
        

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.combined_dim = 16384 + text_dim + num_classes
        self.fc_mu = nn.Linear(self.combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.combined_dim, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim + num_classes, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, audio, text, label_onehot):
        a_feat = self.audio_encoder(audio)
        combined_enc = torch.cat([a_feat, text, label_onehot], dim=1)
        mu, logvar = self.fc_mu(combined_enc), self.fc_logvar(combined_enc)
        z = self.reparameterize(mu, logvar)

        combined_dec = torch.cat([z, label_onehot], dim=1)
        recon = self.decoder(self.decoder_input(combined_dec))
        return recon, mu, logvar   


import torch
import torch.nn as nn

class HardMusicCVAE(nn.Module):
    def __init__(self, latent_dim=32, text_dim=384, num_classes=6):
        super(HardMusicCVAE, self).__init__()
        
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten() 
        )
        
        self.combined_in = 16384 + text_dim + num_classes
        
        self.fc_mu = nn.Linear(self.combined_in, latent_dim)
        self.fc_logvar = nn.Linear(self.combined_in, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim + num_classes, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, audio, text, labels_onehot):
        a_feat = self.audio_encoder(audio) 
        
    
        combined_enc = torch.cat([a_feat, text, labels_onehot], dim=1)      
        
        mu, logvar = self.fc_mu(combined_enc), self.fc_logvar(combined_enc)
        z = self.reparameterize(mu, logvar)
        
        
        combined_dec = torch.cat([z, labels_onehot], dim=1) 
        recon = self.decoder(self.decoder_input(combined_dec))
        return recon, mu, logvar