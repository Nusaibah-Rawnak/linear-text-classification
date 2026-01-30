# Programming Assignment 1 - COMP 550
# Linear Text Classification: Semantics vs. Spelling

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import nltk

# random seed 
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)

# experimentation: preprocessing, feature extraction, and model implementation
def load_data(file0, file1):
    texts = []
    labels = []
    
    # read class 0 samples
    with open(file0, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  
                texts.append(line)
                labels.append(0)
    
    # read class 1 samples
    with open(file1, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  
                texts.append(line)
                labels.append(1)
    
    return texts, labels


def evaluate_model(X_train, y_train, X_test, y_test, model, model_name):
    # 5-fold cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    return cv_mean, cv_std, test_acc

def run_experiments(texts, labels, task_name):
    print(f"TASK: {task_name}")
    
    # 70/30 stratified train-test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(texts, labels, test_size=0.3, 
                                                                  random_state=RANDOM_STATE, stratify=labels)
    
    print(f"Training samples: {len(X_train_text)}")
    print(f"Test samples: {len(X_test_text)}\n")
    
    # preprocessing configurations
    configs = [
        {'name': 'Baseline (no preprocessing)', 'lowercase': False, 'stop_words': None, 'use_tfidf': False},
        {'name': 'Lowercase', 'lowercase': True, 'stop_words': None, 'use_tfidf': False},
        {'name': 'Lowercase + Stopwords', 'lowercase': True, 'stop_words': 'english', 'use_tfidf': False},
        {'name': 'Lowercase + TF-IDF', 'lowercase': True, 'stop_words': None, 'use_tfidf': True},
        {'name': 'Lowercase + Stopwords + TF-IDF', 'lowercase': True, 'stop_words': 'english', 'use_tfidf': True},
    ]
    
    # classifiers
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ('Linear SVM', LinearSVC(max_iter=10000, random_state=RANDOM_STATE))
    ]
    
    # n-gram ranges
    ngram_configs = [('Unigrams', (1, 1))]
    
    # bigrams for alliteration task
    if 'morphphon' in task_name.lower() or 'spelling' in task_name.lower() or 'task b' in task_name.lower():
        ngram_configs.append(('Unigrams + Bigrams', (1, 2)))
    
    # run experiments
    results = []
    
    for ngram_name, ngram_range in ngram_configs:
        print(f"\n\nFeature Type: {ngram_name}")
        print(f"{'-'*50}")
        
        for config in configs:
            print(f"\n  Preprocessing: {config['name']}")
            
            # create vectorizer based on configurations
            if config['use_tfidf']:
                vectorizer = TfidfVectorizer(
                    lowercase=config['lowercase'],
                    stop_words=config['stop_words'],
                    ngram_range=ngram_range
                )
            else:
                vectorizer = CountVectorizer(
                    lowercase=config['lowercase'],
                    stop_words=config['stop_words'],
                    ngram_range=ngram_range
                )
            
            # transform text to features
            X_train = vectorizer.fit_transform(X_train_text)
            X_test = vectorizer.transform(X_test_text)
            
            print(f"    Feature dimensions: {X_train.shape[1]}")
            
            # test each classifier
            for clf_name, clf in classifiers:
                cv_mean, cv_std, test_acc = evaluate_model(
                    X_train, y_train, X_test, y_test, clf, clf_name
                )
                
                print(f"    {clf_name}:")
                print(f"      CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
                print(f"      Test Accuracy: {test_acc:.4f}")
                
                results.append({'ngram': ngram_name, 'preprocessing': config['name'], 'classifier': clf_name,
                'cv_mean': cv_mean, 'cv_std': cv_std, 'test_acc': test_acc})
    
    return results

# TEST CODE 
'''if __name__ == "__main__":
    # test data loading
    texts_test, labels_test = load_data('synsem0.txt', 'synsem1.txt')
    print(f"Loaded {len(texts_test)} samples")
    print(f"Class 0: {labels_test.count(0)} samples")
    print(f"Class 1: {labels_test.count(1)} samples")
    print(f"\nFirst class 0 sample: {texts_test[0]}")
    print(f"First class 1 sample: {texts_test[labels_test.index(1)]}")'''

if __name__ == "__main__":
    print("PA1: Linear Text Classification")
    print("-"*50)
    
    # Task A: Sentiment Classification
    texts_A, labels_A = load_data('synsem0.txt', 'synsem1.txt')
    print(f"Task A: {len(texts_A)} samples loaded")
    
    results_A = run_experiments(texts_A, labels_A, "Task A: Sentiment Classification (Semantics)")
    
    # Task B: Alliteration Classification
    texts_B, labels_B = load_data('morphphon0.txt', 'morphphon1.txt')
    print(f"Task B: {len(texts_B)} samples loaded")
    
    results_B = run_experiments(texts_B, labels_B, "Task B: Alliteration Classification (Spelling)")
    
    # Print summary
    print("\nSummary of Best Results:")
    print(f"Task A: Best test accuracy = {max([r['test_acc'] for r in results_A]):.4f}")
    print(f"Task B: Best test accuracy = {max([r['test_acc'] for r in results_B]):.4f}")