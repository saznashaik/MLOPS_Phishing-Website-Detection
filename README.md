# 🛡️ End-to-End MLOps Pipeline for Phishing Website Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/MLflow-Tracking-orange?style=for-the-badge&logo=mlflow" />
  <img src="https://img.shields.io/badge/DVC-Versioning-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-Container-blue?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/MongoDB-Atlas-green?style=for-the-badge&logo=mongodb" />
  <img src="https://img.shields.io/badge/GitHub_Actions-CI%2FCD-black?style=for-the-badge&logo=github" />
</p>

---

## 📌 Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Why Did We Build This?](#2-why-did-we-build-this)
3. [Why Phishing Detection?](#3-why-phishing-detection)
4. [What Is MLOps and Why Does It Matter?](#4-what-is-mlops-and-why-does-it-matter)
5. [Dataset](#5-dataset)
6. [Complete Technology Stack](#6-complete-technology-stack)
7. [Project Architecture](#7-project-architecture)
8. [Pipeline Walkthrough — Every Step Explained](#8-pipeline-walkthrough--every-step-explained)
9. [Project Folder Structure](#9-project-folder-structure)
10. [How to Run This Project](#10-how-to-run-this-project)
11. [API Documentation](#11-api-documentation)
12. [Model Results](#12-model-results)
13. [CI/CD Pipeline](#13-cicd-pipeline)
14. [What You Learn From This Project](#14-what-you-learn-from-this-project)
15. [Resume Talking Points](#15-resume-talking-points)

---

## 1. What Is This Project?

This is a **complete, production-ready MLOps pipeline** that automatically detects whether a given website is **phishing (dangerous/fake)** or **legitimate (safe)**.

At its core, this is a machine learning project — we train a classification model on real website data and it learns to distinguish dangerous websites from safe ones. But what makes this project special is **not just the model** — it is everything around the model.

In real companies, a data scientist does not just open a Jupyter notebook, train a model, and call it done. That model needs to:
- Be trained on fresh, validated data automatically
- Have its performance tracked and compared across versions
- Be served through an API that other applications can call
- Be packaged so it runs identically on any machine
- Be redeployed automatically every time the code changes

This project does **all of those things**. It is the difference between a school project and something you would actually find running in a real company.

---

## 2. Why Did We Build This?

The gap between "I trained a model" and "I deployed a model" is one of the biggest gaps in data science education. Most courses teach you how to build models in Jupyter notebooks. Almost none teach you what happens after — how do you version your data? How do you track which model version performed best? How do you serve predictions to real users? How do you make sure your pipeline re-runs automatically when something changes?

This project was built specifically to close that gap. It is designed for people who:
- Know basic Python and machine learning
- Want to break into ML Engineer or MLOps Engineer roles
- Need a project that demonstrates real-world engineering practices
- Want to confidently talk about production ML in interviews

Every single tool in this project was chosen because it is **actually used in the industry**. This is not a toy stack. MongoDB, DVC, MLflow, FastAPI, Docker, and GitHub Actions are tools you will encounter in real ML engineering jobs.

---

## 3. Why Phishing Detection?

Phishing attacks are one of the most common forms of cybercrime. A phishing website is a fake website designed to look like a real one — a fake bank login page, a fake shopping site — with the goal of stealing your credentials or money.

We chose phishing detection as the problem for several reasons:

**It is a real, important problem.** According to cybersecurity reports, over 300,000 phishing attacks are reported every month. Companies like Google, Microsoft, and Cloudflare use ML models similar to what we build here to protect their users.

**The dataset is rich and well-structured.** The Kaggle phishing dataset has 87 features extracted from URL properties (like URL length, presence of HTTPS, use of IP address instead of domain name, number of special characters, etc.). These features are meaningful and the model actually learns real patterns.

**It is binary classification** — either phishing (1) or legitimate (0). This is one of the most common problem types in industry, so the skills transfer directly.

**The model performs very well** — achieving around 96-97% accuracy — which makes for a satisfying and convincing demo.

---

## 4. What Is MLOps and Why Does It Matter?

**MLOps** stands for Machine Learning Operations. It is the practice of applying DevOps (software engineering best practices) to machine learning systems.

Think of it this way:

A normal software application has one type of artifact — **code**. When the code changes, you redeploy the app. MLOps systems have three types of artifacts that all need versioning and management:

| Artifact | Changes When | Managed By |
|----------|-------------|------------|
| Code | Developer writes new logic | Git |
| Data | New data is collected | DVC |
| Model | Re-training produces new weights | MLflow + DVC |

Without MLOps practices, teams run into problems like:
- "The model worked on my machine but not on the server"
- "We don't know which version of data was used to train the model in production"
- "Someone changed the code and now we don't know if the model performance dropped"
- "We want to roll back to last month's model but we don't know how"

This project solves all of those problems using industry-standard tools.

---

## 5. Dataset

**Name:** Web Page Phishing Detection Dataset  
**Source:** [Kaggle — Shashwat Work](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)  
**Size:** 11,430 URLs × 87 features  
**Target:** `status` column — `phishing` (1) or `legitimate` (0)  
**Class Balance:** Roughly 50/50, making it ideal for classification without heavy imbalance handling

### What the Features Look Like

The dataset contains features extracted directly from URL structure and web page properties. Some examples:

| Feature | What It Measures | Why It Matters |
|---------|-----------------|----------------|
| `url_length` | Total characters in the URL | Phishing URLs tend to be very long |
| `nb_dots` | Number of dots in URL | Attackers use many subdomains |
| `nb_hyphens` | Number of hyphens | Fake domains often use hyphens |
| `nb_at` | Presence of @ symbol | `@` in URL is a classic phishing trick |
| `has_https` | Whether URL uses HTTPS | Legitimate sites mostly use HTTPS |
| `ip` | Whether URL uses an IP address | Real companies use domain names |
| `domain_length` | Length of the domain part | Short domains are more legitimate |
| `nb_subdomains` | Number of subdomains | Phishing uses many subdomains |
| `page_rank` | PageRank of the domain | Well-known sites have higher rank |

These are the same kinds of features that real browser security systems analyze.

---

## 6. Complete Technology Stack

Every tool was chosen deliberately. Here is what each one is, why we chose it, and how it fits into the pipeline.

---

### 🐍 Python 3.11
**What it is:** The programming language the entire project is written in.  
**Why we use it:** Python is the dominant language in data science and ML engineering. Libraries like scikit-learn, pandas, and MLflow are Python-native.  
**How we use it:** Every script — data ingestion, validation, preprocessing, training, and the API — is written in Python.

---

### 🍃 MongoDB Atlas
**What it is:** A cloud-hosted NoSQL database. Unlike SQL databases (which store data in fixed tables), MongoDB stores data as flexible JSON-like documents.  
**Why we use it:** In real MLOps systems, raw data often comes from multiple sources and gets stored in a database before processing. MongoDB Atlas gives us a free, cloud-based data store that mirrors real production setups. It also means our data is not just a local CSV file — it is centrally stored and accessible from any machine, including GitHub Actions servers.  
**How we use it:**  
- `src/data_ingestion.py` reads the raw CSV and uploads all records to a MongoDB collection called `phishing_db.raw_data`  
- Every downstream step (validation, preprocessing) pulls data from MongoDB instead of directly from the CSV  
- This mimics how real pipelines work — the CSV is just the initial seed; MongoDB is the source of truth

---

### 📦 DVC (Data Version Control)
**What it is:** A version control system for data and ML pipelines, built on top of Git.  
**Why we use it:** Git is excellent for versioning code but completely unsuitable for large data files (CSVs, model weights). DVC tracks data files by storing a small `.dvc` pointer file in Git while keeping the actual large file in a separate storage location. It also lets you define your entire ML pipeline as a series of stages so that running `dvc repro` automatically re-runs only the stages whose dependencies have changed.  
**How we use it:**  
- `dvc add data/raw/dataset_phishing.csv` tells DVC to track the raw data file  
- `dvc.yaml` defines the four-stage pipeline: ingestion → validation → preprocessing → training  
- `dvc repro` runs the full pipeline in the correct order, skipping cached stages  
- If you change only the training script, DVC knows to re-run only the training stage, not re-download and re-preprocess all the data


---

### 🔬 Scikit-learn
**What it is:** The most widely used machine learning library in Python, providing implementations of hundreds of algorithms.  
**Why we use it:** Scikit-learn provides a consistent, clean API for training, evaluating, and saving models. Its `RandomForestClassifier`, `GradientBoostingClassifier`, and `LogisticRegression` are battle-tested algorithms that work well on tabular data like ours.  
**How we use it:**  
- `src/preprocess.py` uses scikit-learn's `LabelEncoder` and `train_test_split`  
- `src/train.py` trains three models from scikit-learn and compares them  
- The best model is saved using `joblib` and loaded by FastAPI at serving time

---

### 📈 MLflow
**What it is:** An open-source platform for managing the full ML lifecycle — experiment tracking, model versioning, and deployment.  
**Why we use it:** Without MLflow, you have no record of which model you trained, with which settings, on which data, and what accuracy it achieved. Two weeks later when someone asks "what was the accuracy of the model we deployed?", you would have no answer. MLflow solves this by automatically recording everything during training. It also provides a browser-based dashboard where you can compare experiments visually.  
**How we use it:**  
- `mlflow.set_experiment("phishing-detection")` creates a named experiment  
- `mlflow.log_params(params)` records hyperparameters like `n_estimators=100`  
- `mlflow.log_metrics(metrics)` records accuracy, precision, recall, F1 score  
- `mlflow.sklearn.log_model(model, ...)` saves the trained model to MLflow's artifact store  
- Run `mlflow ui` to open the dashboard at `http://localhost:5000`

---

### ⚡ FastAPI
**What it is:** A modern, high-performance Python web framework for building REST APIs.  
**Why we use it:** A trained model sitting in a `.pkl` file is useful to nobody except Python scripts. To make the model useful in the real world — for web applications, mobile apps, or other services — it needs to be served through an API. FastAPI is the industry's current favourite for ML APIs because it is extremely fast, automatically generates interactive documentation (Swagger UI), and validates request/response data automatically using Python type hints.  
**Why not Flask?** Flask is older and requires more boilerplate. FastAPI is async-native, validates inputs automatically, and generates docs out of the box. Almost all new ML APIs in industry are built on FastAPI.  
**How we use it:**  
- `app.py` defines three endpoints: `GET /` (health check), `GET /model-info` (show best model metrics), `POST /predict` (make a prediction)  
- The model is loaded once when the server starts, not on every request (much more efficient)  
- `uvicorn app:app --reload` starts the server locally

---

### 🐳 Docker
**What it is:** A platform that packages your application and all its dependencies into a portable container — like a shipping container for software.  
**Why we use it:** The classic problem in software is "it works on my machine." Docker eliminates this completely. A Docker container includes the OS layer, Python version, all installed libraries, and your code — so it runs identically everywhere. This is how every production ML system is deployed.  
**How we use it:**  
- The `Dockerfile` defines a build starting from `python:3.11-slim`  
- It copies `requirements.txt`, installs dependencies, copies all project files, and sets the startup command  
- `docker build -t phishing-detection .` creates the image  
- `docker run -p 8000:8000 --env-file .env phishing-detection` starts the container  
- The GitHub Actions CI/CD pipeline also builds the Docker image to verify the build does not break

---

### ⚙️ GitHub Actions
**What it is:** GitHub's built-in CI/CD (Continuous Integration / Continuous Deployment) system. It runs automated workflows in response to events like pushing code.  
**Why we use it:** Manual deployment is error-prone and slow. CI/CD means every time you push code to GitHub, an automated system checks that your pipeline still works from end to end. If something breaks, you find out immediately instead of when a user reports it.  
**How we use it:**  
- `.github/workflows/ci-cd.yml` defines the workflow  
- Triggered on every push to the `main` branch  
- Steps: install Python → install dependencies → run ingestion → run validation → run preprocessing → train model → test API health → build Docker image  
- The `MONGO_URI` is stored as a GitHub Secret so it is never exposed in the code

---

### 🔐 python-dotenv
**What it is:** A library that loads environment variables from a `.env` file.  
**Why we use it:** Secrets like database passwords must never be hardcoded in code or committed to Git. `.env` files are listed in `.gitignore` so they are never uploaded. On GitHub Actions, the same secrets are provided via GitHub Secrets. `python-dotenv` makes the same code work in both environments without modification.

---

## 7. Project Architecture

```
Raw CSV File
     │
     ▼
MongoDB Atlas ──────────────────── (Cloud data store)
     │
     ▼
Data Ingestion (src/data_ingestion.py)
     │   Reads CSV, pushes to MongoDB
     ▼
Data Validation (src/data_validation.py)
     │   Simple checks + Evidently HTML reports
     │   Fails pipeline if critical issues found
     ▼
Preprocessing (src/preprocess.py)
     │   Drop unneeded columns, encode labels,
     │   fill missing values, train/test split
     │   Saves to data/processed/
     ▼
Model Training (src/train.py)
     │   Trains RandomForest, GradientBoosting, LogisticRegression
     │   Logs all metrics + params to MLflow
     │   Saves best model as models/best_model.pkl
     ▼
FastAPI Server (app.py)
     │   Loads best_model.pkl at startup
     │   Serves POST /predict endpoint
     ▼
Docker Container (Dockerfile)
     │   Packages everything into portable image
     ▼
GitHub Actions (ci-cd.yml)
     │   Runs entire pipeline on every push
     │   Builds and tests Docker image
     ▼
 Production Ready ✅
```

**DVC** sits above all of this, versioning data files and orchestrating the pipeline stages via `dvc.yaml`. Running `dvc repro` triggers the entire flow from ingestion to training automatically.

---

## 8. Pipeline Walkthrough — Every Step Explained

### Stage 1 — Data Ingestion
**File:** `src/data_ingestion.py`

The raw CSV dataset is read into a pandas DataFrame and every row is uploaded to a MongoDB collection. This might seem like an unnecessary step — why not just read the CSV directly? — but it simulates the real world where data arrives continuously from multiple sources into a central database. Downstream stages always pull from MongoDB, making the pipeline independent of where the original data came from.

### Stage 2 — Data Validation
**File:** `src/data_validation.py`

Before any model training happens, we verify the data is actually good. The simple validation checks for missing values, confirms the target column exists, checks for extreme class imbalance, and counts duplicate rows. Evidently AI then generates two detailed HTML reports — a data quality report and a data drift report — saved in `reports/validation/`. If the simple validation fails, the pipeline stops here and does not proceed to training on bad data.

### Stage 3 — Preprocessing
**File:** `src/preprocess.py`

The raw data needs to be transformed before a model can learn from it. We drop columns that are not useful as features (like the raw URL string and MongoDB's internal `_id`), encode text labels to numbers (converting "phishing"/"legitimate" to 1/0), fill any missing values with column medians, and encode any remaining text columns with LabelEncoder. Finally, we split 80% for training and 20% for testing, using stratification to ensure both splits have the same ratio of phishing to legitimate. All four splits are saved as CSV files in `data/processed/`.

### Stage 4 — Model Training
**File:** `src/train.py`

Three models are trained and compared: Random Forest, Gradient Boosting, and Logistic Regression. For each model, MLflow records the hyperparameters, all four evaluation metrics (accuracy, precision, recall, F1), and saves the model as an artifact. A comparison table is printed showing all three models side by side. The best model (by F1 score) is copied to `models/best_model.pkl` and its info is saved in `config/best_model.json`. This "champion model" is what the API loads.

### Stage 5 — API Serving
**File:** `app.py`

FastAPI loads the best model once at server startup. The `/predict` endpoint accepts a JSON body containing a dictionary of features, builds a single-row DataFrame in the correct column order, runs the prediction, and returns the label ("PHISHING" or "LEGITIMATE") along with the confidence percentage. The `/model-info` endpoint exposes the metrics of the currently deployed model, which is useful for monitoring dashboards.

### Stage 6 — Containerisation
**File:** `Dockerfile`

The entire application — Python, all dependencies, all code, and the trained model — is packaged into a Docker image. This image can be run on any Linux server, cloud platform, or colleague's machine and will behave identically. The `.dockerignore` file prevents secrets (`.env`), Git history (`.git`), and MLflow logs (`mlruns/`) from being copied into the image, keeping it lean.

### Stage 7 — CI/CD
**File:** `.github/workflows/ci-cd.yml`

Every push to the `main` branch triggers this workflow on a free GitHub-hosted Ubuntu server. It installs Python, installs all dependencies, recreates the `.env` file from GitHub Secrets, runs every pipeline stage, health-checks the running API, and builds the Docker image. If any step fails, the whole workflow fails and GitHub notifies you immediately. This means broken code can never silently reach production.

---

## 9. Project Folder Structure

```
phishing-mlops/
│
├── src/                          # All pipeline source code
│   ├── __init__.py               # Makes src a Python package
│   ├── data_ingestion.py         # Stage 1: CSV → MongoDB
│   ├── data_validation.py        # Stage 2: Quality checks + reports
│   ├── data_preprocess.py             # Stage 3: Clean + split data
│   ├── data_train.py                  # Stage 4: Train + MLflow logging
│   └── predict.py                # Prediction helper used by API
│
├── data/
│   ├── raw/
│   │   └── dataset_phishing.csv          # Original dataset
│   │   └── dataset_phishing.csv.dvc      # DVC pointer (committed to Git)
│   └── processed/                        # Output of preprocessing
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── best_model.pkl                    # Champion model (loaded by API)
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── logistic_regression.pkl
│
├── reports/
│   ├── validation/                       # Evidently HTML reports
│   └── model_eval/                       # Classification reports per model
│
├── config/
│   └── best_model.json                   # Metrics of the deployed model
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml                     # GitHub Actions pipeline
│
├── mlruns/                               # MLflow experiment data (auto-created)
│
├── app.py                                # FastAPI application
├── dvc.yaml                              # DVC pipeline definition
├── Dockerfile                            # Container definition
├── .dockerignore                         # Files excluded from Docker image
├── .gitignore                            # Files excluded from Git
├── .env                                  # Secrets — NEVER commit this
├── requirements.txt                      # All Python dependencies
└── README.md                             # This file
```

---

## 10. How to Run This Project

### Prerequisites

- Python 3.10 or 3.11 installed
- Git installed
- Docker Desktop installed (for the Docker steps)
- A free MongoDB Atlas account
- A free GitHub account

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/phishing-mlops.git
cd phishing-mlops
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

**3. Install all dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your environment variables**

Create a `.env` file in the project root:
```
MONGO_URI=mongodb+srv://your_username:your_password@cluster0.xxxxx.mongodb.net/
```

**5. Place the dataset**

Download `dataset_phishing.csv` from Kaggle and place it at:
```
data/raw/dataset_phishing.csv
```

**6. Run the complete pipeline**
```bash
dvc repro
```

This single command runs all four stages in order: ingestion → validation → preprocessing → training.

**7. View MLflow experiment results**
```bash
mlflow ui
```
Open `http://localhost:5000` in your browser.

**8. Start the prediction API**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/docs` for the interactive API documentation.

**9. Run with Docker**
```bash
# Build the image
docker build -t phishing-detection .

# Run the container
docker run -p 8000:8000 --env-file .env phishing-detection
```

---

## 11. API Documentation

The API is built with FastAPI and provides automatic interactive documentation at `/docs`.

### Endpoints

#### `GET /`
Health check. Returns API status and whether the model is loaded.

**Response:**
```json
{
  "message": "Phishing Detection API is running!",
  "status": "healthy",
  "model_loaded": true
}
```

#### `GET /health`
Lightweight health check for monitoring systems and Docker health probes.

**Response:**
```json
{
  "status": "ok",
  "model_ready": true
}
```

#### `GET /model-info`
Returns performance metrics of the currently deployed model.

**Response:**
```json
{
  "model_name": "Random Forest",
  "accuracy": 0.9712,
  "precision": 0.9698,
  "recall": 0.9731,
  "f1_score": 0.9714,
  "run_id": "abc123..."
}
```

#### `POST /predict`
The main prediction endpoint. Send a dictionary of URL features and receive a prediction.

**Request body:**
```json
{
  "features": {
    "url_length": 75,
    "nb_dots": 3,
    "nb_hyphens": 2,
    "nb_at": 0,
    "nb_qm": 1,
    "nb_eq": 2,
    "has_https": 1,
    "ip": 0,
    "nb_subdomains": 2,
    "domain_length": 18
  }
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "PHISHING",
  "confidence_percent": 94.3,
  "phishing_probability": 94.3,
  "safe_probability": 5.7
}
```

---

## 12. Model Results

Three models were trained and compared. The best model is automatically selected based on F1 score and saved as `models/best_model.pkl`.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~97.1% | ~96.9% | ~97.3% | ~97.1% |
| Gradient Boosting | ~96.4% | ~96.1% | ~96.7% | ~96.4% |
| Logistic Regression | ~92.1% | ~91.8% | ~92.4% | ~92.1% |

**Why F1 Score is the primary metric:**
Accuracy alone can be misleading when dealing with security problems. A model that is very precise (rarely flags safe sites as phishing) but has low recall (misses many actual phishing sites) would be dangerous. F1 score is the harmonic mean of precision and recall, so it only scores high when both are high. For phishing detection, missing a real phishing site (low recall) is more costly than occasional false alarms (lower precision), so recall is especially important.

---

## 13. CI/CD Pipeline

The GitHub Actions workflow runs automatically on every push to `main`.

```
Push to main
     │
     ▼
GitHub spins up Ubuntu server
     │
     ▼
Install Python 3.11
     │
     ▼
pip install -r requirements.txt
     │
     ▼
Create .env from GitHub Secrets
     │
     ▼
python src/data_ingestion.py       ← Stage 1
     │
     ▼
python src/data_validation.py      ← Stage 2
     │
     ▼
python src/preprocess.py           ← Stage 3
     │
     ▼
python src/train.py                ← Stage 4
     │
     ▼
uvicorn app:app & curl /health     ← API smoke test
     │
     ▼
docker build -t phishing-detection ← Docker build test
     │
     ▼
✅ Pipeline Complete
```

**Setting up GitHub Secrets:**
1. Go to your repository on GitHub
2. Click Settings → Secrets and variables → Actions
3. Click New repository secret
4. Name: `MONGO_URI`, Value: your MongoDB connection string

---

## 14. What You Learn From This Project

By building this project end to end, you develop hands-on experience with:

**Data Engineering**
- Storing and retrieving data from a cloud NoSQL database (MongoDB Atlas)
- Validating data quality programmatically before it enters a pipeline
- Generating professional data quality reports with Evidently AI

**Machine Learning Engineering**
- Building modular, reusable preprocessing pipelines
- Training and comparing multiple model types systematically
- Evaluating models with multiple metrics beyond just accuracy
- Understanding why F1 score matters more than accuracy in imbalanced problems

**MLOps Fundamentals**
- Versioning data files alongside code using DVC
- Defining reproducible pipelines with dependency tracking in `dvc.yaml`
- Tracking experiments so you can always answer "which model is in production and why"
- The difference between a Jupyter notebook model and a production ML system

**Software Engineering for ML**
- Structuring an ML project as proper Python packages with imports
- Using environment variables and secrets management properly
- Writing a REST API that serves ML model predictions
- Containerising an ML application with Docker

**DevOps for ML**
- Understanding CI/CD concepts and why automated pipelines matter
- Writing GitHub Actions workflows that test ML pipelines
- Using GitHub Secrets to manage credentials securely

---

## Dependencies

```
pandas              # Data manipulation
numpy               # Numerical computing
scikit-learn        # Machine learning models and utilities
pymongo             # MongoDB Python driver
dnspython           # Required for MongoDB+SRV connection strings
python-dotenv       # Load .env file for secrets
mlflow              # Experiment tracking and model registry
dvc                 # Data versioning and pipeline orchestration
evidently           # Data validation and drift monitoring reports
fastapi             # REST API framework
uvicorn             # ASGI server to run FastAPI
joblib              # Model serialisation and deserialisation
```

---

## License

This project is open source and available under the MIT License.

---

## Author

Built as a complete MLOps learning project demonstrating production-grade machine learning engineering practices.

If you found this helpful, please ⭐ star the repository.
