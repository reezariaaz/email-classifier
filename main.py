# main.py
import pandas as pd
import os
import glob

from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV


def extract_email_info(file_path):
    """Extract subject, sender, date, and body from an HTML email."""
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


def main():
    # ============================================
    # 1. LOAD LABELS
    # ============================================

    df = pd.read_csv('data/train_labels.csv')

    print("First 5 rows:")
    print(df.head())

    print("\nColumns:", df.columns.tolist())
    print(f"\nTotal rows: {len(df)}")

    print("\nCategories:")
    print(df['true_category'].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    # ============================================
    # 2. EXTRACT ALL TRAINING EMAILS
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
            print(f"Missing file: {filename}")

    train_df = pd.DataFrame(training_data)
    print(f"\nExtracted {len(train_df)} emails")

    train_df['text'] = train_df['subject'] + " " + train_df['body']

    print("\nSample text preview:")
    print(train_df['text'].iloc[0][:300] + "...")

    # ============================================
    # 3. STOPWORDS EXPERIMENT
    # ============================================

    print("\n" + "="*50)
    print("STOPWORDS EXPERIMENT")
    print("="*50)

    X = train_df['text'].values
    y = train_df['category'].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipeline_stopwords = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('clf', LinearSVC(random_state=42))
    ])

    pipeline_no_stopwords = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words=None)),
        ('clf', LinearSVC(random_state=42))
    ])

    acc_stop = cross_val_score(pipeline_stopwords, X, y, cv=cv, scoring='accuracy')
    acc_no_stop = cross_val_score(pipeline_no_stopwords, X, y, cv=cv, scoring='accuracy')

    print(f"With stopwords removed:  {acc_stop.mean():.2%} \u00b1 {acc_stop.std():.2%}")
    print(f"Without stopwords removed: {acc_no_stop.mean():.2%} \u00b1 {acc_no_stop.std():.2%}")

    if acc_no_stop.mean() >= acc_stop.mean():
        best_stop_words = None
        print("Result: Not removing stopwords is better")
    else:
        best_stop_words = 'english'
        print("Result: Removing stopwords is better")

    # ============================================
    # 4. COMPARE MODELS
    # ============================================

    print("\n" + "="*50)
    print("TRAINING CLASSIFIERS (PIPELINE + CV)")
    print("="*50)

    # Map of model name -> a fresh, unfitted classifier instance.
    # Used both to build the comparison pipelines below and, once the
    # winner is picked, to build the final pipelines without repeating
    # ourselves or hard-coding which model "wins".
    candidate_classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(random_state=42)
    }

    models = {
        name: Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words=best_stop_words)),
            ('clf', clf)
        ])
        for name, clf in candidate_classifiers.items()
    }

    results = []

    for name, pipeline in models.items():
        acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro')

        results.append({
            'Model': name,
            'Mean Accuracy': acc_scores.mean(),
            'Std Dev': acc_scores.std(),
            'Macro F1': f1_scores.mean()
        })
        print(f"{name:20} | Accuracy: {acc_scores.mean():.2%} \u00b1 {acc_scores.std():.2%} | F1: {f1_scores.mean():.2%}")

    print("\n" + "-"*60)
    print("MODEL COMPARISON")
    print("-"*60)
    print(f"{'Model':20} | {'Mean Accuracy':15} | {'Std Dev':10} | {'Macro F1':10}")
    print("-"*60)
    for r in results:
        print(f"{r['Model']:20} | {r['Mean Accuracy']:15.2%} | {r['Std Dev']:10.2%} | {r['Macro F1']:10.2%}")

    # Pick the best-performing model from the comparison instead of
    # hard-coding which one "wins" — the experiment decides, not us.
    best_result = max(results, key=lambda r: r['Mean Accuracy'])
    best_model_name = best_result['Model']
    print(f"\nBest model: {best_model_name} "
          f"({best_result['Mean Accuracy']:.2%} \u00b1 {best_result['Std Dev']:.2%} accuracy, "
          f"{best_result['Macro F1']:.2%} macro F1)")

    # ============================================
    # 5. CLASSIFICATION REPORT & CONFUSION MATRIX
    # ============================================

    print("\n" + "="*50)
    print(f"CLASSIFICATION REPORT ({best_model_name})")
    print("="*50)

    best_pipeline = models[best_model_name]

    cv_predictions = cross_val_predict(best_pipeline, X, y, cv=cv)

    print("\nClassification Report:")
    # zero_division=0 avoids a noisy warning if a category ever gets zero
    # predicted samples in a fold (can happen with such a small dataset).
    print(classification_report(y, cv_predictions, zero_division=0))

    print("\nConfusion Matrix:")
    print("Rows: Actual, Columns: Predicted")
    labels = sorted(set(y))
    cm = confusion_matrix(y, cv_predictions, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # ============================================
    # 6. FINAL MODEL WITH CALIBRATED CONFIDENCE
    # ============================================

    print("\n" + "="*50)
    print(f"FINAL MODEL \u2014 {best_model_name} (CALIBRATED)")
    print("="*50)

    # A fresh, unfitted instance of the winning classifier for calibration.
    final_base_classifier = candidate_classifiers[best_model_name]

    calibrated_model = CalibratedClassifierCV(
        estimator=final_base_classifier,
        method='sigmoid',
        cv=5
    )

    calibrated_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words=best_stop_words)),
        ('clf', calibrated_model)
    ])

    calibrated_pipeline.fit(X, y)

    print(f"Trained calibrated model on {len(train_df)} emails")

    test_files = glob.glob('data/test/*.html')
    print(f"\nFound {len(test_files)} test emails")

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

    predictions = calibrated_pipeline.predict(test_texts)
    probabilities = calibrated_pipeline.predict_proba(test_texts)
    confidence_scores = [round(max(p), 4) for p in probabilities]

    results_df = pd.DataFrame({
        'email_id': [d['email_id'] for d in test_data],
        'predicted_category': predictions,
        'confidence_score': confidence_scores
    })

    print(f"\nPredictions complete for {len(results_df)} emails")

    results_df.to_csv('results.csv', index=False)
    print("\nResults saved to results.csv")

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(results_df)


if __name__ == "__main__":
    main()