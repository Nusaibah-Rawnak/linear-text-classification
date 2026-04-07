# Linear Text Classification: Semantics vs. Spelling

A comparative study of linear classifiers for distinguishing semantic features (sentiment) from morphophonological features (alliteration) in text data. Includes a live interactive demo powered by a FastAPI backend.

🔗 [Live Demo](https://nlp-classifier-frontend.vercel.app) · [Full Report](report.pdf)

## Project Overview

This project implements and evaluates **Logistic Regression** and **Linear SVM** classifiers on two distinct text classification tasks:

- **Task A (Semantics)**: Binary sentiment classification (positive vs. negative sentences)
- **Task B (Spelling/Phonology)**: Alliteration detection (alliterative vs. non-alliterative sentences)

The goal is to explore how different preprocessing techniques and feature representations affect classification performance across semantic and spelling-based tasks.

## Results

| Task | Best Config | Test Accuracy |
|------|-------------|---------------|
| Task A — Sentiment | Baseline + LR/SVM | 70.83% |
| Task B — Alliteration | Baseline + LR/SVM | 93.75% |

The 23-percentage-point gap reveals that linear classifiers excel at explicit surface-form patterns (alliteration) but face greater challenges with semantic understanding (sentiment).

## Key Findings

1. **Baseline preprocessing performed best for both tasks** — lowercasing, stopword removal, and TF-IDF all degraded performance
2. **Alliteration is more linearly separable than sentiment** — repeated initial letters create strong, consistent feature signals
3. **Both classifiers perform comparably** — Logistic Regression and Linear SVM achieved identical best accuracies

## Dataset

- `synsem0.txt` — Negative sentiment examples (80 sentences)
- `synsem1.txt` — Positive sentiment examples (80 sentences)
- `morphphon0.txt` — Non-alliterative sentences (80 sentences)
- `morphphon1.txt` — Alliterative sentences (80 sentences)

## Installation

```bash
git clone https://github.com/Nusaibah-Rawnak/linear-text-classification.git
cd linear-text-classification

pip install numpy scikit-learn nltk
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

Run experiments:
```bash
python pa1.py
```

Run the API locally:
```bash
pip install fastapi uvicorn
uvicorn main:app --reload
```

## Project Structure
```
.
├── pa1.py              # Experiment code (preprocessing, training, evaluation)
├── main.py             # FastAPI backend for live demo
├── requirements.txt    # Backend dependencies
├── render.yaml         # Render deployment config
├── report.pdf          # Full research report
├── synsem0.txt         # Negative sentiment data
├── synsem1.txt         # Positive sentiment data
├── morphphon0.txt      # Non-alliterative data
├── morphphon1.txt      # Alliterative data
└── README.md
```

## Tech Stack

- **Research**: Python, scikit-learn, NLTK, NumPy
- **Backend**: FastAPI, uvicorn (deployed on Render)
- **Frontend**: React, TypeScript, Tailwind, Recharts (deployed on Vercel)

## Author

Nusaibah Binte Rawnak  
[LinkedIn](https://linkedin.com/in/nusaibahbinterawnak) | [GitHub](https://github.com/Nusaibah-Rawnak)

## License

This project is available for educational purposes.