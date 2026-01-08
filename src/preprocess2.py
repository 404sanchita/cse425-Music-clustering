import os
import librosa
import numpy as np
import torch

def extract_labeled_spectrograms(base_path, n_mels=128, duration=10):
    features = []
    labels = []
    
    lang_map = {'bangla': 0, 'english': 1, 'hindi': 2, 'korean': 3, 'japanese': 4, 'spanish': 5}
    
    for lang_name, lang_id in lang_map.items():
        folder_path = os.path.join(base_path, lang_name)
        if not os.path.exists(folder_path): continue
        
        print(f"Processing: {lang_name}")
        for file in os.listdir(folder_path):
            if file.endswith(('.mp3', '.wav')):
                try:
                    path = os.path.join(folder_path, file)
                    y, sr = librosa.load(path, duration=duration)
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                    S_DB = librosa.power_to_db(S, ref=np.max)
                    
                
                    if S_DB.shape[1] > 128: S_DB = S_DB[:, :128]
                    elif S_DB.shape[1] < 128: continue
                    S_DB = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min() + 1e-6)
                    
                    features.append(S_DB)
                    labels.append(lang_id)
                except Exception as e:
                    print(f"Error {file}: {e}")
                
    return torch.tensor(np.array(features)).unsqueeze(1), torch.tensor(labels)


base_audio_path = r'C:\Users\sanch\MusicClusteringProject\dataset\audio'
data_2d, labels = extract_labeled_spectrograms(base_audio_path)
torch.save({'data': data_2d, 'labels': labels}, 'processed_data_2d_labeled.pt')
print("Saved labeled data!")