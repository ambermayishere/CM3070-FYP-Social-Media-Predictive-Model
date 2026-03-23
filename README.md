# CM3070-FYP-Social-Media-Predictive-Model
Predictive Modelling of Social Media Trend Emergence — a modular Python project for social media trend analysis using data collection, NLP, feature engineering, network analysis, topic modelling, and machine learning.
## Project Overview

This project analyses social media content to identify signals that may indicate the rise of an emerging trend. It combines data collection, natural language processing, feature engineering, topic modelling, social network analysis, and machine learning into one modular pipeline.

## Features

- Collects social media data using the Twitter/X API or a fallback synthetic dataset.
- Cleans and preprocesses text data.
- Performs sentiment analysis on social media posts.
- Builds time-series feature matrices using 15-minute windows.
- Runs topic modelling to identify major discussion themes.
- Constructs a user interaction network and calculates centrality measures.
- Trains and compares machine learning models for trend prediction.
- Displays results in an interactive Streamlit dashboard.

## Libraries Used

This project is implemented in Python and uses the following libraries and modules:

- `pandas` – data handling and analysis.
- `numpy` – numerical operations.
- `streamlit` – interactive web application.
- `matplotlib` – plots and visualisations.
- `tweepy` – Twitter/X API data collection.
- `TextBlob` – sentiment analysis.
- `networkx` – social network analysis.
- `gensim` – topic modelling with LDA.
- `scikit-learn` – machine learning and model evaluation.
- `pickle` – saving and loading trained models.
- `json` – JSON file handling.
- `os` – file and directory management.
- `re` – text cleaning and preprocessing.
- `warnings` – suppressing non-critical warnings.

## Installation

Install the required Python packages:

```bash
pip install pandas numpy streamlit matplotlib tweepy textblob networkx gensim scikit-learn
```

If you are using a virtual environment, activate it first before installing the dependencies.

## How to Run

### 1. Run the Streamlit application
```bash
streamlit run app.py
```

### 2. Run the data collection module
```bash
python collector.py
```

### 3. Run topic modelling
```bash
python topic_model.py
```

### 4. Run network analysis
```bash
python network.py
```

### 5. Run machine learning evaluation
```bash
python ml_model.py
```

## Project Structure

```text
.
├── app.py
├── collector.py
├── topic_model.py
├── network.py
├── ml_model.py
├── data/
└── README.md
```

## Output Files

The project generates outputs such as:

- `raw_tweets.csv`
- `raw_tweets.json`
- `ml_features.csv`
- `topic_features.csv`
- `network_features.csv`
- `model_results.csv`
- `best_model.pkl`

## Notes

- The collector can use live Twitter/X API access if a bearer token is available.
- If API access is unavailable, the system falls back to synthetic sample data.
- The machine learning labels are currently based on an engagement threshold for evaluation purposes.
- The project is designed as a modular academic system and can be extended for real-time deployment or multi-platform trend analysis.

## Author

Ishita Bhatnagar
