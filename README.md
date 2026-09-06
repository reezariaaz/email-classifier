
# 📧 RedRock Email Classifier

Automated email classification for financial services using NLP and ML.

## Overview
RedRock receives thousands of client emails daily. This system classifies each email into one of five categories for routing to the correct department: **Account Management, Investment Advisory, Loan Processing, Insurance Claims, Other**.

## Dataset
44 training emails / 12 test emails across 5 categories (Account Mgmt: 13, Investment Advisory: 12, Insurance Claims: 7, Other: 6, Loan Processing: 6).

## Approach
1. **Ingestion** — Parse HTML emails with BeautifulSoup; combine subject + body into one text field.
2. **Features** — TF-IDF (unigrams). No stopword removal performed best (100% CV accuracy vs. 95.56% with removal).
3. **Modeling** — Stratified 5-fold CV, leak-free pipeline. Compared Naive Bayes, Logistic Regression, and Linear SVM:

   | Model | Accuracy | Macro F1 |
   |---|---|---|
   | Naive Bayes | 77.50% | 62.11% |
   | Logistic Regression | 77.50% | 61.59% |
   | **Linear SVM (selected)** | **95.56%** | **90.10%** |

4. **Confidence** — CalibratedClassifierCV (Platt scaling); scores range 0.44–0.79.

## Results
12 test emails classified — full predictions in `results.csv`. Macro F1: **0.94**. "Other" had lowest recall (0.67, 2 misclassified); all other categories performed at or near perfect precision/recall.

## Regulatory Considerations
Calibrated confidence scores support a human-review threshold: high confidence → automated routing, low confidence → human review — important given RedRock's regulated environment.

## Future Improvements
More labeled data, conversation history, sender metadata, attachment extraction, human feedback loop.

## How to Run
```bash
cd email-classifier
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Output saved to `results.csv`.

## Project Structure

email-classifier/
├── data/
│ ├── train/ # 44 training emails (.html)
│ ├── test/ # 12 test emails (.html)
│ └── train_labels.csv
├── main.py
├── requirements.txt
├── README.md
└── results.csv


## Requirements
`pandas`, `scikit-learn`, `beautifulsoup4`, `lxml`, `numpy`

---
**Author:** Reeza Damons · **Date:** September 2026