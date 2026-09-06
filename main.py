
# main.py
import pandas as pd

# Load the CSV that tells me which category each training email belongs to
df = pd.read_csv('data/train_labels.csv')

# Quick sanity check - look at the first few rows to make sure it loaded correctly
print("First 5 rows:")
print(df.head())

# Confirm the column names are what I expect
print("\nColumns:", df.columns.tolist())

# Confirm how many training emails I actually have
print(f"\nTotal rows: {len(df)}")

# See how many emails fall into each category - helps me spot class imbalance
print("\nCategories:")
print(df['true_category'].value_counts())

# Check if any rows have missing data before I go further
print("\nMissing values:")
print(df.isnull().sum())

########################################################################################

from bs4 import BeautifulSoup

def extract_email_info(file_path):
    """
    Opens a single email HTML file and pulls out the parts I need:
    the subject, the sender, the date, and the main body text.
    Each email file follows the same layout, so I can rely on
    specific tags/classes to find each piece.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    # The subject is stored in the first <p> tag
    subject = soup.find('p')
    subject = subject.text.strip() if subject else ''

    # Sender and date live inside a div with class "meta"
    # The first line is the sender, the last line is the date
    meta = soup.find('div', class_='meta')
    if meta:
        lines = meta.get_text().strip().split('\n')
        sender = lines[0].strip() if len(lines) > 0 else ''
        date = lines[-1].strip() if len(lines) > 1 else ''
    else:
        sender = ''
        date = ''

    # The actual email content is inside a div with class "email-body"
    body_div = soup.find('div', class_='email-body')
    body = body_div.get_text(separator=' ', strip=True) if body_div else ''

    return subject, sender, date, body

########################################################################################

# main.py 

from bs4 import BeautifulSoup
import os

# Before processing every email, I want to inspect just one file first
# to confirm my assumptions about its structure
email_path = 'data/train/email_1.html'

if os.path.exists(email_path):
    # Read the raw HTML so I can see what I'm working with
    with open(email_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Print the first chunk of the file so I can eyeball the structure
    print("\n" + "="*50)
    print("HTML PREVIEW (first 1000 characters):")
    print("="*50)
    print(html_content[:1000])
    print("\n... (truncated)")
    print("="*50)
    
    # Parse it so I can test what BeautifulSoup is able to pull out
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Check whether the subject can be found in a <title> tag
    subject_tag = soup.find('title')
    if subject_tag:
        print(f"\nSubject found: {subject_tag.get_text(strip=True)}")
    else:
        print("\nNo title/subject found in HTML")
    
    # Preview all the visible text on the page, just to confirm it reads correctly
    body_text = soup.get_text()
    print(f"\nExtracted text preview (first 500 chars):")
    print("-"*50)
    print(body_text[:500].strip())
    print("-"*50)
    
else:
    # If this fails, it usually means the folder structure is wrong
    print(f"File not found: {email_path}")
    print("Check that your training emails are in data/train/")
    
#######################################################################################################

# ============================================
# 4. EXTRACT ALL TRAINING EMAILS
# ============================================

print("\n" + "="*50)
print("EXTRACTING ALL TRAINING EMAILS")
print("="*50)

# I'll build up a list of dictionaries, one per email, then turn it into a DataFrame
training_data = []

for filename in df['filename']:
    file_path = os.path.join('data/train', filename)
    
    if os.path.exists(file_path):
        # Pull out the subject, sender, date, and body for this email
        subject, sender, date, body = extract_email_info(file_path)
        training_data.append({
            'filename': filename,
            'subject': subject,
            'sender': sender,
            'date': date,
            'body': body,
            # Look up the correct label for this email from the labels file
            'category': df[df['filename'] == filename]['true_category'].values[0]
        })
    else:
        # Flag any email listed in the labels file that I can't actually find on disk
        print(f"Missing file: {filename}")

# Turn the list of extracted emails into a proper DataFrame
train_df = pd.DataFrame(training_data)
print(f"\nExtracted {len(train_df)} emails")

# Combine subject and body into one text field - this is what I'll feed the model
train_df['text'] = train_df['subject'] + " " + train_df['body']

print("\nSample text preview:")
print(train_df['text'].iloc[0][:300] + "...")

###############################################################################################

# ============================================
# 5. TF-IDF + CLASSIFIERS (FIXED)
# ============================================

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd

print("\n" + "="*50)
print("TRAINING CLASSIFIERS (PIPELINE + CV)")
print("="*50)

# X is the input text, y is the label I'm trying to predict
X = train_df['text'].values
y = train_df['category'].values

# I'm comparing three different models to see which one classifies best.
# Each pipeline bundles the TF-IDF step together with the classifier so that
# the vectorizer only ever learns from the training fold, not the test fold
# (this avoids data leakage during cross-validation).
models = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', MultinomialNB())
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Linear SVM': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', LinearSVC(random_state=42))
    ])
}

# Since my dataset is small, I use 5-fold stratified cross-validation
# so every category is fairly represented in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, pipeline in models.items():
    # Accuracy tells me the overall correct-prediction rate
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    # Macro F1 tells me how well the model does across all categories equally,
    # including the smaller ones, not just the biggest category
    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro')
    
    results.append({
        'Model': name,
        'Mean Accuracy': f"{acc_scores.mean():.2%}",
        'Std Dev': f"{acc_scores.std():.2%}",
        'Macro F1': f"{f1_scores.mean():.2%}"
    })
    print(f"{name:20} | Accuracy: {acc_scores.mean():.2%} ± {acc_scores.std():.2%} | F1: {f1_scores.mean():.2%}")

# Print a clean side-by-side comparison of all three models
print("\n" + "-"*60)
print("MODEL COMPARISON")
print("-"*60)
print(f"{'Model':20} | {'Mean Accuracy':15} | {'Std Dev':10} | {'Macro F1':10}")
print("-"*60)
for r in results:
    print(f"{r['Model']:20} | {r['Mean Accuracy']:15} | {r['Std Dev']:10} | {r['Macro F1']:10}")
    
###################################################################################################

# ============================================
# 5a. STOPWORDS EXPERIMENT
# ============================================

print("\n" + "="*50)
print("STOPWORDS EXPERIMENT")
print("="*50)

# I want to test whether removing common words (like "the", "and", "is")
# actually helps or hurts the model, since financial emails might use
# those words meaningfully in context

# Version that removes English stopwords
pipeline_stopwords = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LinearSVC(random_state=42))
])

# Version that keeps every word, including stopwords
pipeline_no_stopwords = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words=None)),
    ('clf', LinearSVC(random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_stop = cross_val_score(pipeline_stopwords, X, y, cv=cv, scoring='accuracy')
acc_no_stop = cross_val_score(pipeline_no_stopwords, X, y, cv=cv, scoring='accuracy')

print(f"With stopwords removed:  {acc_stop.mean():.2%} ± {acc_stop.std():.2%}")
print(f"Without stopwords removed: {acc_no_stop.mean():.2%} ± {acc_no_stop.std():.2%}")

# Whichever version scores higher tells me which approach to use going forward
if acc_stop.mean() >= acc_no_stop.mean():
    print("Keeping stopwords removal is better")
else:
    print("Not removing stopwords is better")

###################################################################################################

# ============================================
# 5b. CLASSIFICATION REPORT & CONFUSION MATRIX
# ============================================

print("\n" + "="*50)
print("CLASSIFICATION REPORT (BEST MODEL — LINEAR SVM)")
print("="*50)

# Rebuild the best-performing pipeline so I can dig into exactly
# where it's getting things right or wrong
best_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LinearSVC(random_state=42))
])

# Generate predictions for every email using cross-validation
# (each prediction comes from a fold where that email wasn't used for training)
cv_predictions = cross_val_predict(best_pipeline, X, y, cv=cv)

# This report breaks down precision, recall, and F1 score per category
# so I can see which categories the model struggles with
print("\nClassification Report:")
print(classification_report(y, cv_predictions))

# The confusion matrix shows exactly which categories get mixed up with each other
print("\nConfusion Matrix:")
print("Rows: Actual, Columns: Predicted")
print("Categories:", sorted(set(y)))
cm = confusion_matrix(y, cv_predictions)
print(pd.DataFrame(cm, 
                   index=sorted(set(y)), 
                   columns=sorted(set(y))))

###################################################################################################

# ============================================
# 6. FINAL MODEL + PROPER CONFIDENCE SCORES
# ============================================

from sklearn.calibration import CalibratedClassifierCV
import glob

print("\n" + "="*50)
print("FINAL MODEL — LINEAR SVM (CALIBRATED)")
print("="*50)

# Now that I've picked Linear SVM as the best model, I train it on
# all 44 training emails instead of just a fold, so it learns from everything
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LinearSVC(random_state=42))
])

final_pipeline.fit(X, y)

# A plain LinearSVC doesn't give proper probability scores, only a raw
# decision value. To get real confidence percentages, I wrap it in a
# CalibratedClassifierCV, which uses Platt scaling to convert those
# raw scores into calibrated probabilities.
# Note: newer sklearn versions use 'estimator' instead of 'base_estimator'
calibrated_model = CalibratedClassifierCV(
    estimator=LinearSVC(random_state=42),
    method='sigmoid',  # Platt scaling
    cv=5
)

# Wrap the calibrated model in the same TF-IDF pipeline structure
calibrated_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', calibrated_model)
])

# Fit this final calibrated pipeline on all the training data
calibrated_pipeline.fit(X, y)

print(f"Trained calibrated model on {len(train_df)} emails")

# Find every email in the test set that I need to generate predictions for
test_files = glob.glob('data/test/*.html')
print(f"\nFound {len(test_files)} test emails")

# Extract subject/body text from each test email the same way I did for training
test_data = []
test_texts = []

for file_path in sorted(test_files):
    filename = os.path.basename(file_path)
    subject, sender, date, body = extract_email_info(file_path)
    combined_text = subject + " " + body
    test_data.append({
        'email_id': filename,
        'subject': subject,
        'sender': sender,
        'date': date,
        'body': body,
        'text': combined_text
    })
    test_texts.append(combined_text)

# Predict the category for each test email
predictions = calibrated_pipeline.predict(test_texts)

# Get the calibrated probability for each prediction and use the highest one
# as the model's confidence score for that email
probabilities = calibrated_pipeline.predict_proba(test_texts)
confidence_scores = [round(max(p), 4) for p in probabilities]

# Put everything together into a results table
results_df = pd.DataFrame({
    'email_id': [d['email_id'] for d in test_data],
    'predicted_category': predictions,
    'confidence_score': confidence_scores
})

print(f"\nPredictions complete for {len(results_df)} emails")

# Save the final predictions so they can be reviewed or used downstream
results_df.to_csv('results.csv', index=False)
print("\nResults saved to results.csv")

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(results_df)