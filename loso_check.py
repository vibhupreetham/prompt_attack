import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

merged_df = pd.read_csv("data/merged_corpus.csv")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
vectorizer.fit(merged_df['text'].astype(str))

sources = merged_df['source'].unique()
print("Leave-One-Source-Out Validation Results:")
for held_out in sources:
    train_df = merged_df[merged_df['source'] != held_out]
    test_df = merged_df[merged_df['source'] == held_out]
    
    X_train_loso = vectorizer.transform(train_df['text'].astype(str))
    y_train_loso = train_df['label']
    
    X_test_loso = vectorizer.transform(test_df['text'].astype(str))
    y_test_loso = test_df['label']
    
    clf = XGBClassifier(eval_metric='logloss', base_score=0.5)
    clf.fit(X_train_loso, y_train_loso)
    preds = clf.predict(X_test_loso)
    acc = accuracy_score(y_test_loso, preds)
    try:
        f1 = f1_score(y_test_loso, preds, zero_division=0)
    except:
        f1 = 0
    
    print(f"Held out: {held_out:12s} | Test Size: {len(test_df):5d} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
