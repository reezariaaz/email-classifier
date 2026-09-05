
# main.py
import pandas as pd

# Load the training labels CSV
df = pd.read_csv('data/train_labels.csv')

# Display the first few rows
print("First 5 rows:")
print(df.head())

# Check column names
print("\nColumns:", df.columns.tolist())

# Check number of rows
print(f"\nTotal rows: {len(df)}")

# Check category names
print("\nCategories:")
print(df['true_category'].value_counts())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

########################################################################################

from bs4 import BeautifulSoup

def extract_email_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    subject = soup.find('p')
    subject = subject.text.strip() if subject else ''

    meta = soup.find('div', class_='meta')
    if meta:
        lines = meta.get_text().strip().split('\n')
        sender = lines[0].strip() if len(lines) > 0 else ''
        date = lines[-1].strip() if len(lines) > 1 else ''
    else:
        sender = ''
        date = ''

    body_div = soup.find('div', class_='email-body')
    body = body_div.get_text(separator=' ', strip=True) if body_div else ''

    return subject, sender, date, body

########################################################################################

# main.py 

from bs4 import BeautifulSoup
import os

# Path to one training email
email_path = 'data/train/email_1.html'

# Check if the file exists
if os.path.exists(email_path):
    # Read the HTML file
    with open(email_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Print first 1000 characters to see structure
    print("\n" + "="*50)
    print("HTML PREVIEW (first 1000 characters):")
    print("="*50)
    print(html_content[:1000])
    print("\n... (truncated)")
    print("="*50)
    
    # Parse with BeautifulSoup to see what we can extract
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try to find subject
    subject_tag = soup.find('title')
    if subject_tag:
        print(f"\n📌 Subject found: {subject_tag.get_text(strip=True)}")
    else:
        print("\n⚠️ No title/subject found in HTML")
    
    # Try to find body text
    body_text = soup.get_text()
    print(f"\n📄 Extracted text preview (first 500 chars):")
    print("-"*50)
    print(body_text[:500].strip())
    print("-"*50)
    
else:
    print(f"❌ File not found: {email_path}")
    print("Check that your training emails are in data/train/")
    
#######################################################################################################

# ============================================
# 4. EXTRACT ALL TRAINING EMAILS
# ============================================

print("\n" + "="*50)
print("EXTRACTING ALL TRAINING EMAILS")
print("="*50)

training_data = []

for filename in df['filename']:
    file_path = os.path.join('data/train', filename)
    
    if os.path.exists(file_path):
        subject, sender, date, body = extract_email_info(file_path)
        training_data.append({
            'filename': filename,
            'subject': subject,
            'sender': sender,
            'date': date,
            'body': body,
            'category': df[df['filename'] == filename]['true_category'].values[0]
        })
    else:
        print(f"⚠️ Missing file: {filename}")

# Convert to DataFrame
train_df = pd.DataFrame(training_data)
print(f"\n✅ Extracted {len(train_df)} emails")

# Combine subject and body into one text field
train_df['text'] = train_df['subject'] + " " + train_df['body']

print("\nSample text preview:")
print(train_df['text'].iloc[0][:300] + "...")

###############################################################################################

# ============================================
# 5. TF-IDF + CLASSIFIERS
# ============================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

print("\n" + "="*50)
print("TRAINING CLASSIFIERS")
print("="*50)

# Prepare data
X = train_df['text'].values
y = train_df['category'].values

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

print(f"✅ TF-IDF shape: {X_tfidf.shape}")

# Models to test
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(random_state=42)
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    acc_scores = cross_val_score(model, X_tfidf, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X_tfidf, y, cv=cv, scoring='f1_macro')
    
    results.append({
        'Model': name,
        'Mean Accuracy': f"{acc_scores.mean():.2%}",
        'Std Dev': f"{acc_scores.std():.2%}",
        'Macro F1': f"{f1_scores.mean():.2%}"
    })
    print(f"{name:20} | Accuracy: {acc_scores.mean():.2%} ± {acc_scores.std():.2%} | F1: {f1_scores.mean():.2%}")

# Display results table
print("\n" + "-"*60)
print("MODEL COMPARISON")
print("-"*60)
print(f"{'Model':20} | {'Mean Accuracy':15} | {'Std Dev':10} | {'Macro F1':10}")
print("-"*60)
for r in results:
    print(f"{r['Model']:20} | {r['Mean Accuracy']:15} | {r['Std Dev']:10} | {r['Macro F1']:10}")
    
###################################################################################################

# ============================================
# 6. FINAL MODEL + PREDICTIONS
# ============================================

from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
import glob

print("\n" + "="*50)
print("FINAL MODEL — LINEAR SVM")
print("="*50)

# Train final model on all 44 emails
final_model = LinearSVC(random_state=42)
final_model.fit(X_tfidf, y)

print(f"✅ Trained final model on {len(train_df)} emails")

# Find test emails
test_files = glob.glob('data/test/*.html')
print(f"\n📁 Found {len(test_files)} test emails")

# Extract and predict
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

# Transform test text using the SAME vectorizer
X_test = vectorizer.transform(test_texts)

# Predict categories and confidence scores
predictions = final_model.predict(X_test)
decision_scores = final_model.decision_function(X_test)

# Convert decision scores to confidence scores (0-1)
# Take the max score and scale it to a probability-like value
confidence_scores = []
for scores in decision_scores:
    # For multi-class, take the score of the predicted class and normalise
    # using a simple sigmoid-like scaling
    max_score = max(scores)
    # Scale to 0-1 using a sigmoid function
    confidence = 1 / (1 + np.exp(-max_score / 10))  # Divide by 10 for better spread
    confidence_scores.append(round(confidence, 4))

# Create results DataFrame
results_df = pd.DataFrame({
    'email_id': [d['email_id'] for d in test_data],
    'predicted_category': predictions,
    'confidence_score': confidence_scores
})

print(f"\n✅ Predictions complete for {len(results_df)} emails")

# Save results
results_df.to_csv('results.csv', index=False)
print("\n✅ Results saved to results.csv")

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(results_df)