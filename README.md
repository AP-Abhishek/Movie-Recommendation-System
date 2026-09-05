# Movie Recommendation System - Machine Learning Recommender Application

A Machine Learning web application built using **Python**, **Scikit-learn**, **Pandas**, and **Streamlit** to analyze movie metadata and deliver personalized recommendations based on content similarity.

---

## Project Overview & Application Architecture

The **Movie Recommendation System** leverages Content-Based Filtering to analyze movie characteristics—such as genres, keywords, cast members, directors, and plot overviews—to recommend similar titles based on user selection.

The core application architecture follows an end-to-end Machine Learning data pipeline:

1. **Data Ingestion (`src/data_ingestion.py`):** Automatically downloads and authenticates the TMDB 5000 Movie Dataset from Kaggle API into the raw data directory.
2. **Feature Engineering & Preprocessing (`src/preprocessing.py`):** Cleans text metadata, extracts top cast members and directors, merges keywords and genres into unified tag strings, and outputs `final_movies.csv`.
3. **Similarity Engine (`src/recommender.py`):** Vectorizes text tags using `CountVectorizer` and computes a pairwise Cosine Similarity matrix serialized via `joblib` into `models/similarity_matrix.joblib`.
4. **Interactive Web UI (`main.py`):** Built with Streamlit, featuring dynamic auto-complete selection, custom CSS grid layouts, responsive poster cards, and automatic self-healing asset generation for cloud environments.
5. **Automated Testing Suite (`tests/test_pipeline.py`):** Utilizes `pytest` to validate dataset integrity, preprocessing logic, matrix dimensions, and top-5 recommendation output accuracy.

---

## Core Modules & Features

* **Content-Based Recommendation Engine:** Recommends 5 most similar movies based on multi-attribute cosine similarity scores.
* **Interactive Streamlit Web UI:** Clean, responsive interface with search auto-complete and custom CSS card components.
* **Self-Healing Cloud Pipeline:** Automatically generates missing dataset CSVs and model matrices on app startup (enabling seamless cloud deployment on Streamlit Community Cloud).
* **Automated Data Processing:** Combines genres, keywords, overviews, top 3 actors, and director information into low-dimensional feature tags.
* **Unit Tested Pipeline:** 100% test coverage across raw data loading, data cleaning, vectorization, and recommendation outputs.

---

## Data & Pipeline Details

The system processes movie metadata through structured modules:

| Component / Module | Implementation File | Description & Functionality |
| :--- | :--- | :--- |
| **Data Ingestion** | `src/data_ingestion.py` | Downloads raw TMDB 5000 dataset using Kaggle API |
| **Preprocessing** | `src/preprocessing.py` | Cleans metadata, parses JSON attributes, builds `tags`, produces `final_movies.csv` |
| **Similarity Engine** | `src/recommender.py` | Vectorizes tags using `CountVectorizer` and exports `similarity_matrix.joblib` |
| **Streamlit Interface** | `main.py` | Web interface, custom CSS styling, poster rendering, and recommendation display |
| **Pipeline Tests** | `tests/test_pipeline.py` | Pytest test suite validating data flow, file existence, matrix shapes, and output format |


---

## Tech Stack & Specifications

* **Language:** Python `>=3.13`
* **Package Manager:** `uv` (Fast Python package manager with `pyproject.toml` & `uv.lock`)
* **Machine Learning & Data Processing:** Scikit-learn (`>=1.8.0`), Pandas (`>=2.3.3`), NumPy (`>=2.4.1`), SciPy (`>=1.17.0`), Joblib (`>=1.5.3`)
* **Web Framework:** Streamlit (`>=1.53.1`)
* **Testing Framework:** Pytest (`>=9.0.2`)
* **Dataset Source:** TMDB 5000 Movie Dataset

---

## Environment Setup & Running the Application

Follow these steps to run the application locally on your machine:

### Prerequisites
* **Python 3.13** or higher installed.
* **`uv`** package manager installed (`pip install uv`).

### Step 1: Clone the Repository
```bash
git clone https://github.com/AP-Abhishek/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### Step 2: Install Dependencies
```bash
uv sync
```

### Step 3: Run Unit Tests
```bash
uv run pytest
```

### Step 4: Launch the Streamlit Web Application
```bash
uv run streamlit run main.py
```

---

## Project Structure

```text
Movie-Recommendation-System/
├── data/
│   ├── raw/                           # Raw TMDB 5000 dataset (tmdb_5000_movies.csv, tmdb_5000_credits.csv)
│   └── processed/                     # Preprocessed dataset (final_movies.csv) and poster assets
├── models/                            # Serialized model matrix (similarity_matrix.joblib)
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── data_ingestion.py              # Kaggle API dataset fetcher
│   ├── preprocessing.py               # Data cleaning and feature engineering pipeline
│   ├── recommender.py                 # Vectorizer and Cosine Similarity matrix builder
│   └── favicon.png                    # Application favicon asset
├── tests/
│   └── test_pipeline.py               # Automated pytest suite for end-to-end testing
├── .gitignore                         # Git ignored patterns (models, processed data, venvs, cache)
├── main.py                            # Streamlit web application interface
├── pyproject.toml                     # Project dependencies & tool settings (uv, pytest, pyright)
├── requirements.txt                   # Standard dependency requirements
├── uv.lock                            # Environment lockfile for reproducible builds
└── README.md                          # Comprehensive project documentation
```
