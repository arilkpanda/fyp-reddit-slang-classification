# Reddit Slang Classifier# fyp-reddit-slang-classification

An interactive dashboard for analyzing and classifying Reddit comments using machine learning ensemble methods.

## Overview
This dashboard provides real-time classification of Reddit comments into different slang categories and visualizes linguistic trends from 2014-2024.

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/arilkpanda/fyp-reddit-slang-classification.git
cd fyp-reddit-slang-classification
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app1.py
```

## Required Files
The following files must be present in the same directory as app1.py:
- tfidf_vectorizer.pkl
- tokenizer.pkl
- mlb.pkl
- svm_v2.pkl
- lgbm_v2.pkl
- bilstm_ensemble.keras
- ensemble_model.pkl
- idf_dict.pkl

## Features
- Real-time slang classification
- Comprehensive slang dictionary
- Interactive trend visualizations
- Model performance analysis
- Token-level interpretation