from sentence_transformers import SentenceTransformer
import torch

checkpoint = torch.load('processed_data_2d_labeled.pt')
data = checkpoint['data']
labels = checkpoint['labels']

text_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

languages = ['bangla', 'english', 'hindi', 'korean', 'japanese', 'spanish']
lang_embeddings = text_model.encode(languages) 

text_features = []
for label in labels:
    text_features.append(lang_embeddings[label])

text_features = torch.tensor(text_features, dtype=torch.float32)

torch.save({
    'audio': data,           
    'text': text_features,   
    'labels': labels         
}, 'hybrid_data.pt')

print(f"Hybrid data saved! Audio shape: {data.shape}, Text shape: {text_features.shape}")