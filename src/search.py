import torch
import torch.nn.functional as F
from model import HardMusicCVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('hybrid_data.pt')
audio_data = checkpoint['audio'].to(device)
text_data = checkpoint['text'].to(device)
labels = checkpoint['labels']

model = HardMusicCVAE().to(device)
model.load_state_dict(torch.load('hard_model.pth', map_location=device))
model.eval()

print("Indexing music database...")
with torch.no_grad():
    oh_labels = F.one_hot(labels, num_classes=6).float().to(device)
    _, mu, _ = model(audio_data, text_data, oh_labels)

def search_by_language(lang_id, top_k=3):
    languages = ['Bangla', 'English', 'Hindi', 'Korean', 'Japanese', 'Spanish']
    print(f"\nSearching for top {top_k} matches for language: {languages[lang_id]}")
    
    idx = (labels == lang_id).nonzero(as_tuple=True)[0][0]
    query_vec = mu[idx].unsqueeze(0) 
    
    cos_sim = F.cosine_similarity(query_vec, mu)
    
    scores, indices = torch.topk(cos_sim, k=top_k)
    
    for i in range(top_k):
        match_idx = indices[i].item()
        match_lang = languages[labels[match_idx].item()]
        print(f"Match {i+1}: Song Index {match_idx} (Actual: {match_lang}) | Score: {scores[i].item():.4f}")

search_by_language(0)
search_by_language(5)