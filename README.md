
# 📧 RedRock Email Classifier

Automated email classification for financial services using NLP and ML.

## Overview
RedRock receives thousands of client emails daily. This system classifies each email into one of five categories for routing to the correct department: **Account Management, Investment Advisory, Loan Processing, Insurance Claims, Other**.

## Dataset
44 training emails / 12 test emails across 5 categories (Account Mgmt: 13, Investment Advisory: 12, Insurance Claims: 7, Other: 6, Loan Processing: 6).

## Approach
1. **Ingestion** — Parse HTML emails with BeautifulSoup; combine subject + body into one text field.
2. **Features** — TF-IDF (unigrams). An experiment compared stopword removal vs. none; **no stopword removal performed best** (100.00% ± 0.00% CV accuracy vs. 95.56% ± 5.44% with removal), so the final pipeline uses `stop_words=None`.
3. **Modeling** — Stratified 5-fold CV, with TF-IDF and the classifier bundled in a single scikit-learn `Pipeline` so the vectorizer is refit inside each fold (no data leakage). The best-performing model is selected programmatically rather than assumed. Comparison:

   | Model | Mean Accuracy | Std Dev | Macro F1 |
   |---|---|---|---|
   | Naive Bayes | 75.00% | 8.24% | 58.05% |
   | Logistic Regression | 77.50% | 9.64% | 61.70% |
   | **Linear SVM (selected)** | **100.00%** | **0.00%** | **100.00%** |

4. **Confidence** — `CalibratedClassifierCV` (Platt/sigmoid scaling) on the SVM produces real, calibrated probabilities via `.predict_proba()`, rather than a raw decision score.

## Results

**Cross-validation results (on the 44 labeled training emails):** Linear SVM achieved a perfect 100.00% mean 5-fold CV accuracy (macro F1 = 100.00%), based on out-of-fold predictions, with zero variance across folds.

**Test predictions:** The 12 unlabeled test emails were classified using the final calibrated model and saved to `results.csv`, with a confidence score (0.00–1.00) per email, ranging from 0.4313 to 0.7825. Since these emails have no ground-truth labels, no accuracy figure applies to them — only the model's own confidence estimates.

## A note on generalization
Training emails show consistent, category-indicative vocabulary (e.g. "claim" emails labeled Insurance Claims, "account" emails labeled Account Management), which likely explains the perfect CV accuracy — a real-world, unfiltered dataset would almost certainly be noisier and less lexically clean.

Test-set confidence scores are more moderate (0.43–0.78) and appropriately lower on ambiguous or off-category emails — for example, marketing/event content was correctly routed to "Other" at only ~0.43 confidence, rather than being forced into a real category with false certainty. In at least one case, an email with no literal category keyword ("Retirement Planning Consultation... 401(k) rollover") was still correctly classified as Investment Advisory, suggesting the model captures some genuine semantic signal beyond exact keyword matching.

Even so, the small, lexically consistent training set (44 emails) limits how confidently these results generalize to RedRock's real, larger email volume in production. More varied, labeled data would be needed for a reliable real-world performance estimate.

## Regulatory Considerations
Calibrated confidence scores support a human-review threshold: high confidence → automated routing, low confidence → human review — important given RedRock's regulated environment. A confidence score of, say, 0.79 means the calibrated model estimates roughly a 79% probability for the predicted category — it is an estimate, not a guarantee of correctness, which is part of why low-confidence emails should still be routed to a person.

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
```
email-classifier/
├── data/
│   ├── train/          # 44 training emails (.html)
│   ├── test/            # 12 test emails (.html)
│   └── train_labels.csv
├── main.py
├── requirements.txt
├── README.md
└── results.csv
```

## Requirements
`pandas`, `scikit-learn`, `beautifulsoup4`, `lxml`

---
**Author:** Reeza Damons · **Date:** September 2026