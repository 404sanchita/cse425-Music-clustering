import os
import librosa
import numpy as np
import torch

def extract_features(base_path, n_mfcc=13, duration=30):
    features = []
    labels = []
    lang_map = {'bangla': 0, 'english': 1, 'hindi': 2, 'korean': 3, 'japanese': 4, 'spanish': 5}
    
    
    for lang_name, lang_id in lang_map.items():
        lang_folder = os.path.join(base_path, lang_name)
        if not os.path.exists(lang_folder):
            print(f"Warning: Language folder '{lang_folder}' not found. Skipping.")
            continue
        
        print(f"Processing {lang_name}...")
        file_count = 0
        for file in os.listdir(lang_folder):
            if file.endswith(('.mp3', '.wav')):
                try:
                
                    path = os.path.join(lang_folder, file)
                    y, sr = librosa.load(path, duration=duration)
                    
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                    
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    
                    features.append(mfcc_mean)
                    labels.append(lang_id)
                    file_count += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        print(f"  Processed {file_count} {lang_name} files")
    
    if len(features) == 0:
        return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.long)
    
    return torch.tensor(np.array(features), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


audio_path = r'C:\Users\sanch\MusicClusteringProject\dataset\audio'

features, labels = extract_features(audio_path)

if features.shape[0] > 0:
    torch.save(features, 'processed_data.pt')
    print(f"\nDone! Saved {features.shape[0]} songs to processed_data.pt")
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
else:
    print("No songs found. Check your folder path!")
    print(f"Expected structure: {audio_path}/[lang_name]/[audio_files.wav]")