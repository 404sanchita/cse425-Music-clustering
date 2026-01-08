# Music Clustering with Variational Autoencoders

An unsupervised learning pipeline for clustering hybrid language music tracks using Variational Autoencoders (VAE). This project explores different VAE architectures for extracting latent representations from audio and lyrics data, performing clustering, and comparing results with baseline methods.

## Project Overview

This project implements three levels of complexity:

- **Easy Task**: Basic VAE with MFCC features, K-Means clustering, and PCA baseline comparison
- **Medium Task**: Convolutional VAE with hybrid audio+lyrics features, multiple clustering algorithms
- **Hard Task**: Conditional VAE (CVAE) with multi-modal clustering and comprehensive evaluation

## Repository Structure

```
MusicClusteringProject/
├── dataset/                  # Dataset files
│   ├── audio/               # Audio files (organized by language)
|       |──bangla
|       |──english
|       |──hindi
|       |──korean
|       |──japanese
|       |──spanish
├── src/                     # Source code
│   ├── model.py             # VAE model definitions
│   ├── preprocess.py        # MFCC feature extraction (Easy task)
│   ├── preprocess2.py       # Spectrogram extraction (Medium/Hard tasks)
│   ├── hybrid_data.py       # Hybrid feature preparation
│   ├── train.py             # Easy task training script
│   ├── train_medium.py      # Medium task training script
│   ├── train_hard.py        # Hard task training script
│   ├── clustering.py        # Clustering algorithms module
│   ├── evaluation.py        # Evaluation metrics module
│   └── search.py            # Cross-modal retrieval (Hard task)
├── notebooks/               # Jupyter notebooks
│   └── exploratory.ipynb    # Exploratory data analysis
├── results/                 # Results and outputs
│   ├── latent_visualization/  # Visualization plots
│   └── clustering_metrics.csv # Evaluation metrics
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Option 1: Using Provided Dataset Structure

If you have audio files organized by language folders (`dataset/audio/`), run:

```bash
# For Easy Task (MFCC features)
python src/preprocess.py

# For Medium/Hard Tasks (Spectrograms)
python src/preprocess2.py

# For Hybrid Features (Audio + Lyrics)
python src/hybrid_data.py
```

### Option 2: Using External Datasets

You can use datasets like:
- Million Song Dataset (MSD)
- GTZAN Genre Collection
- Jamendo Dataset
- MIR-1K Dataset

Ensure your audio files are organized by language/genre in the `dataset/audio/` directory.

## Usage

### Easy Task

Train a basic VAE and perform clustering:

```bash
python src/train.py
```

This will:
- Train a VAE on MFCC features
- Perform K-Means clustering on latent representations
- Generate t-SNE visualizations
- Compare with PCA + K-Means baseline
- Calculate Silhouette Score and Calinski-Harabasz Index

### Medium Task

Train a convolutional VAE with hybrid features:

```bash
python src/train_medium.py
```

This will:
- Train a ConvVAE on spectrograms with lyrics embeddings
- Perform multiple clustering algorithms (K-Means, Agglomerative, DBSCAN)
- Generate comprehensive visualizations
- Calculate multiple evaluation metrics (Silhouette, ARI, Davies-Bouldin)

**Prerequisites**: Run `src/preprocess2.py` and `src/hybrid_data.py` first.

### Hard Task

Train a Conditional VAE for multi-modal clustering:

```bash
python src/train_hard.py
```

This will:
- Train a CVAE with audio, lyrics, and label conditioning
- Perform comprehensive clustering evaluation
- Generate reconstruction examples
- Calculate all metrics including NMI and Cluster Purity
- Compare with multiple baselines

**Prerequisites**: 
- Run `src/preprocess2.py` and `src/hybrid_data.py`
- Ensure `hard_model.pth` is saved after training

### Cross-Modal Retrieval (Hard Task)

After training the Hard Task model, you can perform cross-modal search:

```bash
python src/search.py
```

## Evaluation Metrics

The project evaluates clustering quality using:

- **Silhouette Score**: Measures cluster cohesion and separation
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Index**: Average similarity between clusters (lower is better)
- **Adjusted Rand Index (ARI)**: Agreement with ground truth labels
- **Normalized Mutual Information (NMI)**: Mutual information between clusters and labels
- **Cluster Purity**: Fraction of dominant class in each cluster

All metrics are saved to `results/clustering_metrics.csv`.

## Models

### MusicVAE (Easy Task)
- Basic feedforward VAE for MFCC features
- Encoder: Linear layers → Latent space (μ, σ)
- Decoder: Latent space → Reconstructed features

### ConvVAE (Medium Task)
- Convolutional VAE for spectrograms
- Encoder: Conv2D layers → Latent space
- Decoder: ConvTranspose2D layers → Reconstructed spectrograms

### HybridConvVAE (Medium Task)
- Convolutional VAE with multi-modal input
- Audio encoder: Conv2D layers
- Text encoder: Pre-trained SentenceTransformer embeddings
- Fuses audio and lyrics in latent space

### HardMusicCVAE (Hard Task)
- Conditional VAE with label conditioning
- Conditions on language labels during encoding and decoding
- Enables controlled generation and better disentanglement

## Results

Results are saved in the `results/` directory:
- Visualization plots in `results/latent_visualization/`
- Evaluation metrics in `results/clustering_metrics.csv`
- Trained models: `hard_model.pth`, `medium_model.pth` (if saved)

## Requirements

See `requirements.txt` for complete list. Key dependencies:

- PyTorch
- scikit-learn
- librosa
- matplotlib
- numpy
- sentence-transformers
- umap-learn (optional, for UMAP visualization)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in training scripts
2. **File not found errors**: Ensure data preprocessing scripts have been run
3. **Import errors**: Verify all dependencies are installed: `pip install -r requirements.txt`
4. **Empty results folder**: Scripts will create necessary directories automatically

### Data Path Issues

If you encounter path errors, update the paths in:
- `src/preprocess.py`: Update `audio_path`
- `src/train.py`: Update `processed_data.pt` path if needed

## Project Deliverables

1. GitHub Repository with organized code
2. Implementation of Easy, Medium, and Hard tasks
3. Comprehensive evaluation metrics
4. Visualization scripts
5. NeurIPS-style paper report 

## License

This project is for educational/research purposes.

## Contact

For questions or issues, please open an issue in the repository.

