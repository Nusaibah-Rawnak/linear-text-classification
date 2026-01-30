# Linear Text Classification: Semantics vs. Spelling

A comparative study of linear classifiers for distinguishing semantic features (sentiment) from morphophonological features (alliteration) in text data.

## Project Overview

This project implements and evaluates **Logistic Regression** and **Linear SVM** classifiers on two distinct text classification tasks:

- **Task A (Semantics)**: Binary sentiment classification (positive vs. negative reviews)
- **Task B (Spelling/Phonology)**: Alliteration detection (alliterative vs. non-alliterative sentences)

The goal is to explore how different preprocessing techniques and feature representations affect classification performance across semantic and spelling-based tasks.

## Features

### Preprocessing Techniques
- Lowercase normalization
- Stopword removal (English)
- TF-IDF weighting
- N-gram extraction (unigrams and bigrams)

### Classifiers
- **Logistic Regression**: Probabilistic linear classifier with L2 regularization
- **Linear SVM**: Maximum-margin linear classifier

### Evaluation
- 70/30 stratified train-test split
- 5-fold cross-validation on training set
- Accuracy metrics reported for both CV and test sets

## Dataset

The project uses four text files:

- `synsem0.txt` - Negative sentiment examples (80 sentences)
- `synsem1.txt` - Positive sentiment examples (80 sentences)
- `morphphon0.txt` - Non-alliterative sentences (80 sentences)
- `morphphon1.txt` - Alliterative sentences (80 sentences)

## Installation
```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/REPO-NAME.git
cd REPO-NAME

# Install required packages
pip install numpy scikit-learn nltk

# Download NLTK stopwords (run in Python)
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

Run the main script to execute both classification tasks:
```bash
python pa1.py
```

The script will:
1. Load both datasets
2. Run experiments with different preprocessing configurations
3. Test both classifiers (Logistic Regression and Linear SVM)
4. Report cross-validation and test accuracy for each configuration
5. Print a summary of best results

## Results Summary

The experiments reveal that:
- **Task A (Sentiment)**: Unigrams with basic preprocessing achieve strong performance, as sentiment is captured by individual word semantics
- **Task B (Alliteration)**: Bigrams with TF-IDF provide optimal performance, as alliteration patterns require capturing adjacent word relationships

## Project Structure
```
.
├── pa1.py              # Main implementation file
├── synsem0.txt         # Negative sentiment data
├── synsem1.txt         # Positive sentiment data
├── morphphon0.txt      # Non-alliterative data
├── morphphon1.txt      # Alliterative data
└── README.md           # This file
```

## Key Findings

1. **Feature importance varies by task**: Bigrams are crucial for spelling-based tasks but less important for semantic tasks
2. **Preprocessing impact differs**: Stopword removal helps sentiment classification but may hurt alliteration detection
3. **Both classifiers perform comparably**: Logistic Regression and Linear SVM achieve similar accuracy across both tasks

## Dependencies

- Python 3.7+
- NumPy
- scikit-learn
- NLTK


## Author

Nusaibah Binte Rawnak  
[LinkedIn](https://linkedin.com/in/nusaibahbinterawnak) | [GitHub](https://github.com/Nusaibah-Rawnak)

## License

This project is available for educational purposes.
