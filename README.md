# 🛡️ Cost-Sensitive DistilBERT for Toxic Comment Detection

## 📌 Project Overview
This repository contains the official codebase for **NLP Assessment 3: Project Development**. 
Our project tackles the real-world challenge of cyberbullying by developing an automated toxic comment detection system. We explicitly address the widespread issue of **Extreme Class Imbalance** in social media data through a Cost-Sensitive Learning approach.

## 🛠️ Methodology & Tech Stack
- **Architecture:** `distilbert-base-uncased` (Leveraging Self-Attention for context understanding).
- **Baseline Model:** TF-IDF + Logistic Regression.
- **Innovation:** Implementation of **Weighted Cross-Entropy Loss** to apply higher penalty weights to the minority class (Toxic), significantly improving recall.
- **Frameworks:** `PyTorch`, `Hugging Face Transformers`, `Scikit-Learn`.

## 📊 Dataset
We utilized the `SetFit/toxic_conversations` dataset (Parquet format) from Hugging Face. To simulate resource-constrained environments (e.g., Google Colab Free Tier), we sampled:
- **Training Set:** 5,000 samples
- **Test Set:** 1,000 samples

## 🏆 Experiment Results & Trade-off Analysis
Our cost-sensitive DistilBERT model significantly outperformed the traditional TF-IDF baseline, especially in capturing elusive toxic comments (False Negatives).

| Model | Accuracy | Toxic Precision | Toxic Recall | Toxic F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + LR)** | ~89% | High | **Extremely Low** | Poor |
| **Cost-Sensitive DistilBERT** | **91%** | 0.51 | **0.69** | **0.59** |

*Note: As detailed in our comprehensive report, the slight drop in overall accuracy is a deliberate and necessary trade-off to achieve a much higher Recall for the toxic class, which is critical for real-world content moderation.*

## 🚀 How to Run
This project is fully optimized for Google Colab (T4 GPU).
1. Clone this repository or download `NLP_A3_Toxic_Detection.ipynb`.
2. Upload the notebook to Google Colab.
3. Select `Runtime -> Change runtime type -> T4 GPU`.
4. Run all cells sequentially.

---
*Developed for University of Technology Sydney (UTS) NLP Assessment 3.*
