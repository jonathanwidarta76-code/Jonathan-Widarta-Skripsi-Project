# Project Title

Pendekatan Hybrid Machine Learning untuk Prediksi Arah Pergerakan Harga Saham Berbasis Analisis Sentimen Twitter terhadap Saham NVIDIA

📄 Thesis: [Laporan Skripsi — Universitas Multimedia Nusantara, 2026]

🚀 Live Dashboard: Sentiment Signal Dashboard

---

## 📋 Overview
This project investigates whether public sentiment expressed on Twitter/X can be used as a predictive signal for NVIDIA (NVDA) stock price direction. Daily tweet sentiment scores are extracted using VADER (Valence Aware Dictionary and sEntiment Reasoner), engineered into lag-based time-series features, and fed into a hybrid machine learning pipeline that combines LSTM as a temporal feature extractor with a Random Forest classifier.

- Background: Social media sentiment data from Twitter reflects real-time public opinion that is spontaneous and unfiltered compared to news sources. Research shows that sentiment signals from social media typically require 1–2 days before being absorbed into actual market action.
- Objective: Build and compare standalone and hybrid ML models for predicting next-day NVIDIA stock price direction (up/down) using sentiment-derived features.
- Dataset: Secondary dataset from Kaggle — NVIDIA tweets and NVDA historical stock price data.
- Methods: VADER sentiment scoring → daily aggregation → lag feature engineering → model training (SVM, RF, LSTM, Hybrid LSTM+SVM, Hybrid LSTM+RF) → evaluation (accuracy, precision, recall, F1-score, confusion matrix).
- Best result: Hybrid LSTM + Random Forest — accuracy 0.89, F1-score 0.91 on the Sentiment + Lag feature scenario.
- Deployment: Static web dashboard published on GitHub Pages showing real-time sentiment trends and model predictions.


### Key Features
- Hybrid Architecture: LSTM acts as a temporal feature extractor; its 16-neuron dense feature layer output is fed into a Random Forest classifier — combining sequential pattern capture with ensemble robustness.
- Sentiment Feature Engineering: Daily VADER compound scores aggregated from raw tweets with lag features (Lag-1, Lag-2, Lag-3) to capture temporal sentiment dependencies.
- Threshold-based Label Design: Binary price direction label (1 = up, 0 = down) determined by a 0.1% return threshold, justified against NVIDIA's historical daily volatility.
- Underfitting/Overfitting Analysis: Training vs. validation loss reported for each neural model to diagnose convergence behavior.
- Interactive Dashboard: Publicly accessible GitHub Pages dashboard with price prediction visualization, sentiment trend charts, volume/sentiment correlation, and sample tweet panels.

---

## 🎯 Key Contributions

1. Hybrid Sentiment-to-Price Pipeline: A validated end-to-end pipeline from raw tweet text → VADER scoring → lag feature engineering → Hybrid LSTM+RF classification, proven to outperform all standalone baselines on NVDA.
2. Threshold Ablation Study: Empirical comparison of 0.1% threshold vs. no-threshold label schemes, grounded in NVDA's actual daily return volatility distribution.
Convergence Diagnosis: Systematic training/validation loss analysis across LSTM tunggal and Hybrid LSTM backbone, identifying underfitting as the primary failure mode for standalone LSTM on small daily datasets.
3. Hyperparameter Comparison Table: Documented comparison of LSTM units (16/32/64), dropout rates (0.1/0.2/0.3), and feature dense layer neurons (8/16/32/64) — providing evidence-based justification for the final architecture.
Proof-of-Concept Deployment: A fully static, server-free dashboard deployed on GitHub Pages demonstrating the research output as a functional decision-support tool.

---

## Project Architecture / Research Workflow
The research follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework:



## 🚀 Installation

### Requirements
Python 3.8+
TensorFlow 2.x
scikit-learn
pandas, numpy
nltk (for VADER)
joblib

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jonathanwidarta76-code/Jonathan-Widarta-Skripsi-Project.git
cd Jonathan-Widarta-Skripsi-Project
```

2. Install dependencies:
```bash
pip install tensorflow scikit-learn pandas numpy nltk joblib
```

3. Download NLTK resources :
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

## 📊 Dataset Preparation

### Name of The Dataset

Dataset Overview :
A. NVIDIA Tweets ( Source ： Kaggle )
B. NVIDIA Stock Price ( Source : Kaggle )

This project uses two secondary datasets obtained from Kaggle. No direct scraping via Twitter API was performed.

⚠️ Because secondary datasets are used, the date range and tweet volume are fixed to the availability of the Kaggle source. This is an explicit limitation of the study.

### Download

**The dataset can be obtained at the link**

Dataset links:
A. NVIDIA Tweets Dataset — Kaggle : https://www.kaggle.com/datasets/soheiltehranipour/100k-nvidia-tweets
B. NVDA Stock Price Dataset — Kaggle : https://www.kaggle.com/datasets/programmerrdai/nvidia-stock-historical-data

Once downloaded, organize the data as follows:
```
Jonathan-Widarta-Skripsi-Project/
├── Nvidia-Tweets.csv          ← tweet dataset from Kaggle
├── NVDA (1).csv               ← stock price dataset from Kaggle
├── train_model_fixed.py       ← training + export script
├── index.html                 ← dashboard
└── data.json                  ← generated after training (do not edit manually)

```
---

## 🏋️ Training

**Section Description** For big data, explain how to train the model. What to include:
- Training scripts.
- Training parameters.
- Hyperparameters.
- Execution commands.
- GPU/CPU requirements.

Example Structure:
```bash
python train.py --config configs/train.yaml
```

---

## 📊 Results

**Section Description** This section presents the final outcomes, outputs, or achievements of the project. The content of this section may vary depending on the project type.  The goal is to demonstrate what has been successfully developed, implemented, evaluated, or achieved.
What to include:
- Final project outcomes.
- System implementation results.
- Experimental or testing results.
- Visualization results like evaluation graphs, screenshots of the system or dashboard.
- Performance analysis.
- User testing or usability results.
- Comparison with baseline or previous systems or benchmark comparison.
- Deployment results.

You can use table or visualization. 

---

## 🏗️ Project Structure

**Section Description** Explain you project structure.

For Example: 

```
FaceGCD/
├── data_loader/              # Dataset loaders and augmentations
│   ├── augmentations/        # Data augmentation strategies
│   ├── youtube_faces_*.py    # YTF dataset loaders
│   ├── casia_webface_*.py    # CASIA dataset loaders
│   └── data_loaders.py       # Main data loading utilities
├── model/                    # Model architectures
│   ├── dino_vision_transformer.py  # DINO ViT backbone
│   ├── prefix_generator.py   # HyperNetwork-based prefix generator
│   ├── ViT_face.py           # Face-specific ViT components
│   └── mobilenet.py          # Landmark CNN
├── trainer/                  # Training utilities
│   ├── trainer.py            # Main training loop
│   └── faster_mix_k_means_pytorch.py  # Semi-supervised K-Means
├── utils/                    # Utility functions
│   ├── cluster_utils.py      # Clustering utilities
│   ├── losses.py             # Loss functions
│   └── dino_utils.py         # DINO-specific utilities
├── shell/                    # Shell scripts for experiments
├── train.py                  # Training script
├── extract_features.py       # Feature extraction script
├── SSK.py                    # Semi-supervised K-Means evaluation
└── requirements.txt          # Python dependencies
```

---

## 📝 Citation

**Section Description** Use this section if the project is associated with a publication or research paper. What to include:
- Citation format.
- DOI.
- BibTeX entry.

For Example:
If you find this work useful for your research, please cite:

```bibtex
@article{oh2025facegcd,
  title={...},
  authors={...},
  journal={...},
  year={...}
}
```

---

## 🙏 Acknowledgments

**Section Description** Provide acknowledgements to contributors or supporting organizations. What to include:
- (Mandatory) Big Data Lab, Information Systems Study Program, Universitas Multimedia Nusantara (UMN).
- (if any) Sponsors/Partners.
- (if any) Research groups.
- (if any) Dataset providers.
- (if any) Collaborators.

---

## 📧 Contact

**Section Description** Provide maintainer or author contact information.

For Example:
For questions or issues, please:
- Open an issue on GitHub
- Contact: youremail@mail.com

---

## 📜 License

**Section Description** Please state your project license of any. For example:

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Tips

You can just download this Read Me template and modify it. 
---
