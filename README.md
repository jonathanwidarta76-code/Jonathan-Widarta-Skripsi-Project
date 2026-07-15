# HYBRID MACHINE LEARNING APPROACH TO PREDICTION OF STOCK PRICE MOVEMENT DIRECTION BASED ON TWITTER SENTIMENT ANALYSIS TOWARDS NVIDIA STOCK

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

<img width="989" height="885" alt="image" src="https://github.com/user-attachments/assets/14bbb59c-8668-4174-9bf1-43342f3a94bc" />


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
1. NVIDIA Tweets ( Source ： Kaggle )
2. NVIDIA Stock Price ( Source : Kaggle )

This project uses two secondary datasets obtained from Kaggle. No direct scraping via Twitter API was performed.

⚠️ Because secondary datasets are used, the date range and tweet volume are fixed to the availability of the Kaggle source. This is an explicit limitation of the study.

### Download

**The dataset can be obtained at the link**

Dataset links:
1. NVIDIA Tweets Dataset — Kaggle : https://www.kaggle.com/datasets/soheiltehranipour/100k-nvidia-tweets
2. NVDA Stock Price Dataset — Kaggle : https://www.kaggle.com/datasets/programmerrdai/nvidia-stock-historical-data

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

Run the full pipeline (preprocessing → training → evaluation → dashboard export):

Example Structure:
```bash
python train_model_fixed.py
```
This will:
1. Preprocess and clean tweet text
2. Score sentiment with VADER
3. Aggregate to daily sentiment features + lag features
4. Train the Hybrid LSTM + Random Forest model
5. Evaluate on the test set (80/20 time-based split)
6. Save model artifacts:
- hybrid_lstm_model.keras
- rf_hybrid.joblib
- scaler_lstm.joblib
7. Export data.json for the dashboard

---

## 📊 Results

Model Comparison (Best Feature Scenario: Sentiment + Lag)
| No | Model | Accuracy | Precision | Recall | F1-Score |
|----|-------|----------|-----------|--------|----------|
| 1 | **Hybrid Machine Learning + RF** | **0.88** | **0.85** | **1.00** | **0.92** |
| 2 | Hybrid Machine Learning + SVM | 0.77 | 1.00 | 0.66 | 0.80 |
| 3 | SVM | 0.50 | 0.75 | 0.42 | 0.54 |
| 4 | Random Forest | 0.40 | 0.66 | 0.28 | 0.40 |
| 5 | LSTM | 0.33 | 0.00 | 0.00 | 0.00 |

<img width="1886" height="940" alt="image" src="https://github.com/user-attachments/assets/8473aad0-31f3-47cf-977e-c27ddcd8f7e6" />

The live dashboard visualizes:
1. Model accuracy, precision, recall, F1 metrics
2. NVDA closing price with ▲/▼ prediction markers (colored by correctness)
3. Daily sentiment trend (positive/negative/neutral volume + avg compound line)
4. Sentiment mix donut chart (total dataset distribution)
5. Volume & sentiment vs. price direction comparison chart
6. Sample positive and negative tweets by VADER compound score
7. Prediction log table (filterable and sortable)

🌐 View live dashboard
https://jonathanwidarta76-code.github.io/Jonathan-Widarta-Skripsi-Project/

---

## 🏗️ Project Structure
```
Jonathan-Widarta-Skripsi-Project/
├── Nvidia-Tweets.csv              # Raw tweet dataset (Kaggle)
├── NVDA (1).csv                   # Raw stock price dataset (Kaggle)
├── train_model_fixed.py           # Full pipeline: preprocessing → training → export
├── index.html                     # Static dashboard (GitHub Pages)
├── data.json                      # Dashboard data export (generated by training)
├── hybrid_lstm_model.keras        # Saved LSTM feature extractor (generated)
├── rf_hybrid.joblib               # Saved Random Forest classifier (generated)
└── scaler_lstm.joblib             # Saved MinMaxScaler (generated)
```

---

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@thesis{widarta2026hybrid,
  title   = {Pendekatan Hybrid Machine Learning untuk Prediksi Arah Pergerakan
             Harga Saham Berbasis Analisis Sentimen Twitter terhadap Saham NVIDIA},
  author  = {Widarta, Jonathan},
  school  = {Universitas Multimedia Nusantara},
  year    = {2026},
  type    = {Undergraduate Thesis}

```

---

## 🙏 Acknowledgments

1. Big Data Lab, Information Systems Study Program, Universitas Multimedia Nusantara (UMN) — for research support and facilities.
2. Dataset providers on Kaggle — NVIDIA Tweets and NVDA stock price datasets.

---

## 📧 Contact

For questions or issues, please:
Open an issue on GitHub
Contact: jonathan.widarta1@student.umn.ac.id

---

## 📜 License

This project is released for academic purposes under the Universitas Multimedia Nusantara thesis guidelines. See LICENSE file for details.

---
