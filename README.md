

# 📧 RedRock Email Classifier

Automated email classification for financial services using Natural Language Processing and Machine Learning.

---

## 📁 Project Overview

RedRock is a financial services company that receives thousands of client emails daily. This system automatically classifies incoming emails into one of five business categories so they can be forwarded to the correct department.

### Business Categories

1. **Account Management**
2. **Investment Advisory**
3. **Loan Processing**
4. **Insurance Claims**
5. **Other**

---

## 📊 Dataset

| Split | Count |
|-------|-------|
| Training Emails | 44 |
| Test Emails | 12 |
| Categories | 5 |

### Training Category Distribution

| Category | Count |
|----------|-------|
| Account Management | 13 |
| Investment Advisory | 12 |
| Insurance Claims | 7 |
| Other | 6 |
| Loan Processing | 6 |

---

## 🧠 Approach

### 1. Data Ingestion
- Read HTML emails using BeautifulSoup
- Extract: subject, sender, date, email body
- Combine subject + body into a single text field

### 2. Feature Engineering
- TF-IDF vectorization (unigrams only)
- 357 features from 44 training emails

### 3. Model Training & Evaluation
- **Stratified 5-fold cross-validation**
- Compared 3 models:

| Model | Mean Accuracy | Macro F1 |
|-------|---------------|----------|
| Naive Bayes | 75.00% | 57.44% |
| Logistic Regression | 77.50% | 61.59% |
| **Linear SVM** | **91.11%** | **84.15%** |

✅ **Linear SVM** was selected as the final model due to its superior performance and stability.

### 4. Prediction
- 12 test emails processed through the same pipeline
- Confidence scores generated for every prediction

---

## 📈 Results

| email_id | predicted_category | confidence_score |
|----------|-------------------|------------------|
| email_1.html | Investment Advisory | 0.4985 |
| email_10.html | Investment Advisory | 0.5196 |
| email_11.html | Loan Processing | 0.5179 |
| email_12.html | Account Management | 0.5100 |
| email_2.html | Investment Advisory | 0.5057 |
| email_3.html | Investment Advisory | 0.5083 |
| email_4.html | Account Management | 0.5173 |
| email_5.html | Insurance Claims | 0.5176 |
| email_6.html | Account Management | 0.4938 |
| email_7.html | Account Management | 0.5056 |
| email_8.html | Loan Processing | 0.5111 |
| email_9.html | Investment Advisory | 0.5053 |

---

## 🚀 How to Run

### 1. Clone or navigate to the project folder

```bash
cd email-classifier