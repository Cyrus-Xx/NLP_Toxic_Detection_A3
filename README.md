# 🛡️ Cost-Sensitive DistilBERT for Toxic Comment Detection

## 📌 Project Overview
This repository contains the official codebase for **NLP Assessment 3: Project Development**. 
Our project tackles the real-world challenge of cyberbullying by developing an automated toxic comment detection system. We explicitly address the widespread issue of **Extreme Class Imbalance** in social media data through a Cost-Sensitive Learning approach. 

**Highlight:** This project explicitly demonstrates why *Accuracy* alone is a highly misleading metric under severe class imbalance, and why optimizing for *Recall* is critical for real-world moderation.

## 📁 Repository Structure
```text
├── NLP_A3_Toxic_Detection.ipynb  # Main Google Colab notebook (Code & Output)
├── confusion_matrix.png          # Visual evaluation of the DistilBERT model
├── roc_curve.png                 # ROC-AUC visualization
└── README.md                     # Project documentation
```

## 🛠️ Methodology & Tech Stack
- **Architecture:** `distilbert-base-uncased` (Transformer Encoder).
- **Baseline Model:** TF-IDF + Logistic Regression.
- **Innovation:** Implementation of **Weighted Cross-Entropy Loss** to apply higher penalty weights to the minority class (Toxic), forcing the model to prioritize underrepresented data.
- **Frameworks:** `PyTorch`, `Hugging Face Transformers`, `Scikit-Learn`.

## 📊 Dataset
We utilized the `SetFit/toxic_conversations` dataset from Hugging Face. To simulate resource-constrained environments (e.g., Google Colab Free Tier) and adhere to assignment guidelines, we sampled:
- **Training Set:** 5,000 samples
- **Test Set:** 1,000 samples

## 🏆 Experiment Results & The "Accuracy Illusion"
Our cost-sensitive DistilBERT model significantly outperformed the traditional TF-IDF baseline, especially in detecting minority-class toxic comments (minimizing False Negatives).

| Model | Accuracy | Toxic Precision | Toxic Recall | Toxic F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + LR)** | **0.91** | 1.00 | **0.01** | 0.02 |
| **Cost-Sensitive DistilBERT** | **0.91** | 0.51 | **0.69** | **0.59** |

**Trade-off Analysis:** 
Although both models achieved identical overall accuracy (91%), the baseline model almost completely failed to detect toxic comments (Recall = 0.01), essentially acting as a "majority-class guesser." Our cost-sensitive DistilBERT dramatically improved minority-class recall to **0.69**. This perfectly illustrates the *Accuracy Illusion* in imbalanced datasets, proving our proposed solution to be substantially more practical and reliable for real-world content moderation systems.

## 🚀 How to Run
This project is fully optimized for a Google Colab T4 GPU environment.
1. Download `NLP_A3_Toxic_Detection.ipynb` from this repository.
2. Upload the notebook to Google Colab.
3. Navigate to `Runtime -> Change runtime type` and select **T4 GPU**.
4. Run all cells sequentially.

---
*Developed for Natural Language Processing (NLP) Assessment 3.*  
*University of Technology Sydney (UTS), 2026.*
