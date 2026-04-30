# 🛡️ End-to-End MLOps Pipeline for Phishing Website Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/MLflow-Tracking-orange?style=for-the-badge&logo=mlflow" />
  <img src="https://img.shields.io/badge/DVC-Versioning-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-Container-blue?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/MongoDB-Atlas-green?style=for-the-badge&logo=mongodb" />
  <img src="https://img.shields.io/badge/GitHub_Actions-CI%2FCD-black?style=for-the-badge&logo=github" />
  <img src="https://img.shields.io/badge/Docker_Hub-Registry-blue?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/Render-Deployment-46E3B7?style=for-the-badge&logo=render" />
  <img src="https://img.shields.io/badge/Evidently-Monitoring-FF6B6B?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Live%20%26%20Deployed-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Cost-100%25%20Free-gold?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Self--Healing-Auto%20Retrain-blueviolet?style=for-the-badge" />
</p>

---

## 🌐 Live Demo

**API is deployed and running live on Render (free tier):**

| Endpoint | Link |
|----------|------|
| 🏠 API Home | `https://your-app-name.onrender.com/` |
| 📖 Interactive Docs (Swagger UI) | `https://your-app-name.onrender.com/docs` |
| ❤️ Health Check | `https://your-app-name.onrender.com/health` |
| 📊 Model Info | `https://your-app-name.onrender.com/model-info` |
| 🔮 Predict | `POST https://your-app-name.onrender.com/predict` |

> **Note:** Hosted on Render free tier. First request after 15 minutes of inactivity takes ~30 seconds to wake up (cold start). This is expected behaviour on the free plan and would not occur on a paid tier.

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
9. [Cloud Deployment — Docker Hub + Render](#9-cloud-deployment--docker-hub--render)
10. [Production Monitoring and Auto-Retraining](#10-production-monitoring-and-auto-retraining)
11. [Project Folder Structure](#11-project-folder-structure)
12. [How to Run This Project](#12-how-to-run-this-project)
13. [API Documentation](#13-api-documentation)
14. [Model Results](#14-model-results)
15. [CI/CD Pipeline](#15-cicd-pipeline)
16. [Free Tools Used — Zero Cost Stack](#16-free-tools-used--zero-cost-stack)
17. [What You Learn From This Project](#17-what-you-learn-from-this-project)

---

## 1. What Is This Project?

This is a **complete, production-ready, self-healing MLOps pipeline** that automatically detects whether a given website is **phishing (dangerous/fake)** or **legitimate (safe)** — and keeps itself up to date without any human intervention.

At its core, this is a machine learning project — we train a classification model on real website data and it learns to distinguish dangerous websites from safe ones. But what makes this project special is **not just the model** — it is everything around the model.

In real companies, a data scientist does not just open a Jupyter notebook, train a model, and call it done. That model needs to:
- Be trained on fresh, validated data automatically
- Have its performance tracked and compared across versions
- Be served through an API that other applications can call
- Be packaged so it runs identically on any machine
- Be deployed to a real cloud server accessible by anyone
- Be monitored for data drift in production
- Automatically retrain and redeploy when the data changes

This project does **all of those things** — entirely on free services.

---

## 2. Why Did We Build This?

The gap between "I trained a model" and "I deployed a self-monitoring model" is one of the biggest gaps in data science education. Most courses teach you how to build models in Jupyter notebooks. Almost none teach you what happens after — how do you version your data? How do you track which model version performed best? How do you serve predictions to real users? How do you detect when your model starts degrading?

This project was built specifically to close that gap. Every single tool was chosen because it is **actually used in the industry**. This is not a toy stack — MongoDB, DVC, MLflow, FastAPI, Docker, Docker Hub, Render, Evidently, and GitHub Actions are tools you will encounter in real ML engineering jobs.

---

## 3. Why Phishing Detection?

Phishing attacks are one of the most common forms of cybercrime. A phishing website is a fake website designed to look like a real one — a fake bank login page, a fake shopping site — with the goal of stealing your credentials or money.

We chose phishing detection for several reasons:

**It is a real, important problem.** Over 300,000 phishing attacks are reported every month. Companies like Google, Microsoft, and Cloudflare use ML models similar to what we build here to protect their users.

**The dataset is rich and well-structured.** The Kaggle phishing dataset has 87 features extracted from URL properties (like URL length, presence of HTTPS, use of IP address instead of domain name, number of special characters, etc.).

**It is binary classification** — either phishing (1) or legitimate (0). This is one of the most common problem types in industry, so the skills transfer directly.

**The model performs very well** — achieving around 96-97% accuracy — which makes for a convincing demo.

---

## 4. What Is MLOps and Why Does It Matter?

**MLOps** stands for Machine Learning Operations. It is the practice of applying DevOps (software engineering best practices) to machine learning systems.

MLOps systems have three types of artifacts that all need versioning and management:

| Artifact | Changes When | Managed By |
|----------|-------------|------------|
| Code | Developer writes new logic | Git + GitHub Actions |
| Data | New data is collected | DVC + MongoDB Atlas |
| Model | Re-training produces new weights | MLflow + DVC |

Without MLOps practices, teams run into problems like:
- "The model worked on my machine but not on the server"
- "We don't know which version of data was used to train the model in production"
- "Someone changed the code and now we don't know if the model performance dropped"
- "The data has changed over time but we never noticed and the model is making wrong predictions"

This project solves all of those problems using industry-standard, free-tier tools.

---

## 5. Dataset

**Name:** Web Page Phishing Detection Dataset
**Source:** [Kaggle — Shashwat Work](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)
**Size:** 11,430 URLs × 87 features
**Target:** `status` column — `phishing` (1) or `legitimate` (0)
**Class Balance:** Roughly 50/50, making it ideal for classification without heavy imbalance handling

### What the Features Look Like

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

---

## 6. Complete Technology Stack

Every tool was chosen deliberately. Here is what each one is, why we chose it, and how it fits into the pipeline.

---

### 🐍 Python 3.11
**What it is:** The programming language the entire project is written in.
**Why we use it:** Python is the dominant language in data science and ML engineering.
**How we use it:** Every script — data ingestion, validation, preprocessing, training, monitoring, and the API — is written in Python.

---

### 🍃 MongoDB Atlas
**What it is:** A cloud-hosted NoSQL database that stores data as flexible JSON-like documents.
**Why we use it:** In real MLOps systems, raw data is stored in a central database, not local CSV files. MongoDB Atlas is free for small projects and accessible from anywhere — including GitHub Actions runners.
**How we use it:**
- `src/data_ingestion.py` reads the raw CSV and uploads all records to a MongoDB collection called `phishing_db.raw_data`
- Every downstream step pulls from MongoDB, making the pipeline independent of the original file location
- This mimics how real pipelines work — the CSV is just the initial seed; MongoDB is the source of truth

---

### 📦 DVC (Data Version Control)
**What it is:** A version control system for data and ML pipelines, built on top of Git.
**Why we use it:** Git cannot handle large data files. DVC tracks data files using small pointer files while keeping the actual data separate. It also defines the entire pipeline as stages so `dvc repro` re-runs only what changed.
**How we use it:**
- `dvc add data/raw/dataset_phishing.csv` tracks the raw data
- `dvc.yaml` defines the four-stage pipeline: ingestion → validation → preprocessing → training
- `dvc repro` runs the full pipeline, skipping cached stages automatically

---

### 📊 Evidently AI
**What it is:** An open-source library for ML data validation and production monitoring.
**Why we use it:** Evidently generates beautiful HTML reports showing data quality metrics and — critically — **data drift reports** that detect when production data starts looking different from training data. This is what powers our auto-retraining system.
**How we use it:**
- `src/data_validation.py` uses `DataQualityPreset` and `DataDriftPreset` during pipeline validation
- `src/monitor.py` uses `DatasetDriftMetric` in production to measure how much the incoming data has drifted
- Reports saved as HTML files can be opened in any browser
- If drift exceeds 30% of features, the monitoring script exits with code 1, triggering the auto-retrain job in GitHub Actions

---

### 🔬 Scikit-learn
**What it is:** The most widely used machine learning library in Python.
**Why we use it:** Provides consistent, well-tested implementations of classification algorithms.
**How we use it:**
- `src/data_preprocess.py` uses `LabelEncoder` and `train_test_split`
- `src/data_train.py` trains `RandomForestClassifier`, `GradientBoostingClassifier`, and `LogisticRegression`
- Best model saved with `joblib` and loaded by FastAPI at serving time

---

### 📈 MLflow
**What it is:** An open-source platform for ML experiment tracking and model management.
**Why we use it:** Without MLflow, there is no record of which model was trained with which settings and achieved which accuracy. MLflow records everything automatically and provides a browser dashboard to compare experiments.
**How we use it:**
- `mlflow.set_experiment("phishing-detection")` creates a named experiment
- `mlflow.log_params(params)` records hyperparameters
- `mlflow.log_metrics(metrics)` records accuracy, precision, recall, F1
- `mlflow.sklearn.log_model(model, ...)` saves the model as an artifact
- Run `mlflow ui --workers 1` to open the dashboard at `http://localhost:5000`

---

### ⚡ FastAPI
**What it is:** A modern, high-performance Python web framework for building REST APIs.
**Why not Flask?** Flask requires more boilerplate, is not async-native, and does not generate documentation automatically. FastAPI validates inputs automatically, generates Swagger UI docs out of the box, and is what most new ML APIs in industry use.
**How we use it:**
- `app.py` defines four endpoints: `/` (home), `/health` (health check), `/model-info` (deployed model metrics), `/predict` (prediction)
- The model loads once at server startup, not on every request
- `uvicorn app:app --reload` starts the server locally

---

### 🐳 Docker
**What it is:** A platform that packages your application and all its dependencies into a portable container.
**Why we use it:** Docker eliminates "it works on my machine." A Docker container includes the OS layer, Python version, all installed libraries, the trained model, and your code — running identically everywhere.
**How we use it:**
- `Dockerfile` builds from `python:3.11-slim`
- `docker build -t phishing-detection .` creates the image locally
- `docker run -p 8000:8000 --env-file .env phishing-detection` runs it locally
- GitHub Actions builds and pushes the image to Docker Hub automatically on every push

---

### 🐋 Docker Hub
**What it is:** A free public registry for storing and sharing Docker images — like GitHub but for containers.
**Why we use it:** Render needs to pull our Docker image from somewhere. Docker Hub is free for public repositories and integrates perfectly with GitHub Actions. Every time we push code or retrain a model, a new image is built and pushed here.
**How we use it:**
- GitHub Actions builds the Docker image after training completes
- Image is tagged with both the Git commit SHA (`abc1234`) and `latest`
- Render always pulls the `latest` tag when redeploying
- Image URL format: `yourdockerhubusername/phishing-detection:latest`

---

### 🚀 Render.com
**What it is:** A cloud platform with a genuinely free tier for running web services, including Docker containers.
**Why we use it instead of AWS/GCP/Azure:** Those platforms require credit cards and can incur unexpected charges. Render's free tier gives a real public HTTPS URL, runs Docker containers directly, and redeploys automatically via a deploy hook URL — no complex configuration needed.
**How we use it:**
- Render is configured to run our Docker image from Docker Hub
- A **Deploy Hook URL** (secret webhook) lets GitHub Actions trigger a redeploy with one `curl` command
- The environment variable `MONGO_URI` is set directly in Render's dashboard
- After every successful CI/CD run, the new image is deployed automatically
- Your API gets a permanent public URL like `https://phishing-detection.onrender.com`

> **Free tier note:** The container spins down after 15 minutes of no traffic. The first request after inactivity takes ~30 seconds (cold start). In production you would use a paid plan or a keep-alive ping.

---

### ⚙️ GitHub Actions
**What it is:** GitHub's built-in CI/CD system that runs automated workflows on every code push.
**Why we use it:** Manual deployment is error-prone and slow. GitHub Actions runs the entire pipeline — training, testing, building Docker, deploying — automatically on every push. It also runs our nightly monitoring job on a cron schedule.
**How we use it:**
- `ci-cd.yml` — triggered on every push to `main`, runs full ML pipeline, builds Docker image, pushes to Docker Hub, triggers Render deploy
- `monitoring.yml` — runs every night at 2 AM UTC via cron, checks for data drift, auto-retrains if needed, redeploys new model
- All secrets (MONGO_URI, DOCKERHUB_TOKEN, RENDER_DEPLOY_HOOK, etc.) stored in GitHub Secrets — never in code

---

### 🔐 python-dotenv
**What it is:** A library that loads environment variables from a `.env` file.
**Why we use it:** Secrets like database passwords must never be hardcoded in code. `.env` is in `.gitignore` so it is never uploaded. On GitHub Actions, the same secrets come from GitHub Secrets. `python-dotenv` makes the same code work in both environments.

---

## 7. Project Architecture

```
Raw CSV File
     │
     ▼
MongoDB Atlas ──────────────────────────── (Free cloud data store)
     │
     ▼
Data Ingestion (src/data_ingestion.py)
     │   Reads CSV, pushes all rows to MongoDB
     ▼
Data Validation (src/data_validation.py)
     │   Simple checks + Evidently HTML quality reports
     │   Pipeline stops here if critical issues found
     ▼
Preprocessing (src/data_preprocess.py)
     │   Drop unneeded columns, encode labels,
     │   fill missing values, 80/20 train/test split
     │   Saves splits to data/processed/
     ▼
Model Training (src/data_train.py)
     │   Trains RandomForest, GradientBoosting, LogisticRegression
     │   All metrics + params logged to MLflow
     │   Best model saved as models/best_model.pkl
     ▼
FastAPI Server (app.py)
     │   Loads best_model.pkl at startup
     │   Serves /predict, /health, /model-info endpoints
     ▼
Docker Image (Dockerfile)
     │   Packages Python + dependencies + code + model
     │   Built by GitHub Actions
     ▼
Docker Hub (free registry)
     │   Stores the built image
     │   Tagged with commit SHA + "latest"
     ▼
Render.com (free hosting)
     │   Pulls "latest" image from Docker Hub
     │   Runs container, assigns public HTTPS URL
     ▼
🌐 Live API — https://your-app.onrender.com
     │
     ▼
Nightly Monitoring (GitHub Actions cron — 2 AM UTC)
     │   Loads training data as reference
     │   Simulates/collects production data
     │   Runs Evidently drift detection
     │
   ┌─┴──────────────────┐
   │                    │
No drift            Drift > 30%
   │                    │
   ▼                    ▼
Sleep until       Auto-Retrain triggered
tomorrow          src/retrain.py runs full pipeline
                  New best_model.pkl produced
                  New Docker image built + pushed
                  Render redeployed with new model
                  Loop continues tomorrow ↺
```

**DVC** sits above all of this, versioning data files and orchestrating pipeline stages via `dvc.yaml`.

---

## 8. Pipeline Walkthrough — Every Step Explained

### Stage 1 — Data Ingestion
**File:** `src/data_ingestion.py`

The raw CSV dataset is read into a pandas DataFrame and every row is uploaded to a MongoDB collection. This simulates the real world where data arrives continuously from multiple sources into a central database. Downstream stages always pull from MongoDB, making the pipeline independent of where the original data came from.

### Stage 2 — Data Validation
**File:** `src/data_validation.py`

Before any model training happens, we verify the data is actually good. Simple validation checks for missing values, confirms the target column exists, checks for class imbalance, and counts duplicate rows. Evidently AI then generates two HTML reports — a data quality report and a data drift report — saved in `reports/validation/`. If simple validation fails, the pipeline stops here.

### Stage 3 — Preprocessing
**File:** `src/data_preprocess.py`

We drop columns not useful as features (raw URL string, MongoDB `_id`), encode text labels to numbers ("phishing"/"legitimate" → 1/0), fill missing values with column medians, and encode remaining text columns. We split 80% for training and 20% for testing with stratification. All four splits are saved as CSV files in `data/processed/`.

### Stage 4 — Model Training
**File:** `src/data_train.py`

Three models are trained and compared. For each, MLflow records hyperparameters and all four evaluation metrics (accuracy, precision, recall, F1). A comparison table is printed. The best model by F1 score is copied to `models/best_model.pkl` and its info saved in `config/best_model.json`.

### Stage 5 — API Serving
**File:** `app.py`

FastAPI loads the best model once at server startup. The `/predict` endpoint accepts a JSON body of features, builds a single-row DataFrame, runs the prediction, and returns the label and confidence. The `/model-info` endpoint exposes currently deployed model metrics.

### Stage 6 — Containerisation
**File:** `Dockerfile`

The entire application — Python, all dependencies, all code, and the trained model — is packaged into a Docker image. `.dockerignore` prevents secrets (`.env`), Git history, and MLflow logs from entering the image.

### Stage 7 — CI/CD and Cloud Deployment
**Files:** `.github/workflows/ci-cd.yml`

Every push to `main` triggers three jobs: (1) run the full ML pipeline and train the model, (2) build the Docker image and push to Docker Hub, (3) call the Render deploy hook to pull the new image and go live. The API is accessible at a public HTTPS URL within minutes of pushing code.

### Stage 8 — Production Monitoring
**File:** `src/monitor.py`

Every night at 2 AM UTC, GitHub Actions runs the monitoring workflow. It loads the training data as a reference baseline and compares it against simulated production data using Evidently's `DatasetDriftMetric`. If more than 30% of features have drifted, it writes a `drift_status.json` file and exits with code 1, which triggers the auto-retrain job.

### Stage 9 — Auto-Retraining
**File:** `src/retrain.py`

When drift is detected, this script re-runs the full pipeline: fresh data ingestion from MongoDB, preprocessing, and training. The new best model replaces the old one. A retrain history log is saved to `reports/monitoring/retrain_history.json`. After retraining, a new Docker image is built and pushed to Docker Hub, and Render is redeployed automatically — zero human intervention required.

---

## 9. Cloud Deployment — Docker Hub + Render

### Why This Stack Instead of AWS

| Feature | AWS (ECS + ECR) | Our Stack (Docker Hub + Render) |
|---------|----------------|--------------------------------|
| Cost | Can incur charges | 100% free |
| Setup complexity | High (IAM, VPC, clusters) | Simple (GUI-based) |
| Docker image storage | ECR (paid) | Docker Hub (free public repo) |
| Container hosting | ECS Fargate | Render free tier |
| Public HTTPS URL | Requires load balancer setup | Automatic on Render |
| Auto-deploy on push | Requires CodePipeline setup | Deploy hook URL (one curl command) |
| Good for | Production enterprise systems | Portfolio projects, MVPs |

### Deployment Flow

```
GitHub Actions builds Docker image
          │
          ▼
Image pushed to Docker Hub
(yourdockerhubusername/phishing-detection:latest)
          │
          ▼
GitHub Actions calls Render Deploy Hook URL
(one curl command — no AWS config needed)
          │
          ▼
Render pulls latest image from Docker Hub
          │
          ▼
New container starts with updated model
          │
          ▼
https://your-app.onrender.com is live ✅
```

### GitHub Secrets Required for Deployment

| Secret Name | What It Contains |
|-------------|-----------------|
| `MONGO_URI` | MongoDB Atlas connection string |
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (not password) |
| `RENDER_DEPLOY_HOOK` | Render webhook URL from service Settings |
| `RENDER_APP_URL` | Your Render app URL for health checks |

### How to Get the Render Deploy Hook

1. Go to your Render service dashboard
2. Click **Settings** tab
3. Scroll to **Deploy Hook** section
4. Copy the URL — it looks like `https://api.render.com/deploy/srv-xxxxx?key=yyyyy`
5. Add it as `RENDER_DEPLOY_HOOK` in GitHub Secrets

---

## 10. Production Monitoring and Auto-Retraining

### The Problem This Solves

A model trained today on today's data may become inaccurate over time. Phishing websites evolve — attackers find new tricks, change URL patterns, and create new structures. This is called **data drift** — the statistical properties of incoming data shift away from the training distribution.

Without monitoring, you would never know your model is degrading until users start reporting wrong predictions.

### How Our Monitoring Works

```
Every night at 2 AM UTC (GitHub Actions cron)
          │
          ▼
Load X_train.csv as reference baseline
          │
          ▼
Load/simulate production data
(replace simulate_production_data() with real data source)
          │
          ▼
Evidently DatasetDriftMetric runs
Compares feature distributions: reference vs current
          │
          ▼
Drift score calculated
(what % of features have statistically drifted)
          │
     ┌────┴────────────────────┐
     │                         │
Drift ≤ 30%               Drift > 30%
     │                         │
     ▼                         ▼
Save report              Save drift_status.json
No action needed         Exit code 1
                              │
                              ▼
                    GitHub Actions detects exit code 1
                    Triggers auto-retrain job
                              │
                              ▼
                    src/retrain.py runs:
                    1. Fresh data from MongoDB
                    2. Full preprocessing
                    3. Train all 3 models again
                    4. New best model saved
                              │
                              ▼
                    New Docker image built
                    Pushed to Docker Hub
                    Render redeployed
                              │
                              ▼
                    Retrain logged to
                    reports/monitoring/retrain_history.json
```

### Drift Detection Configuration

In `src/monitor.py`:

```python
DRIFT_THRESHOLD = 0.3   # Retrain if >30% of features drift
```

You can make this stricter (0.1 = retrain if 10% drift) or more lenient (0.5 = only retrain on severe drift) depending on how sensitive your use case is.

### Replacing Simulation with Real Production Data

Currently `simulate_production_data()` adds Gaussian noise to training data to simulate drift. In a real production system, replace this function with your actual data source:

```python
def get_production_data():
    # Option 1: Read recent API requests logged to MongoDB
    client = MongoClient(MONGO_URI)
    recent = client["phishing_db"]["api_requests"].find(
        {"timestamp": {"$gte": last_week}}
    )
    return pd.DataFrame(list(recent))

    # Option 2: Read from a data warehouse query
    # Option 3: Read from a message queue (Kafka, SQS)
```

### Manual Retrain Trigger

You can force a retrain at any time without waiting for the nightly schedule:

1. Go to your GitHub repo → **Actions** tab
2. Click **Nightly Model Monitoring and Auto-Retrain**
3. Click **Run workflow**
4. Set `force_retrain` to `true`
5. Click **Run workflow**

This is useful after adding new training data or changing the model code.

---

## 11. Project Folder Structure

```
phishing-mlops/
│
├── src/                               # All pipeline source code
│   ├── __init__.py                    # Makes src a Python package
│   ├── data_ingestion.py              # Stage 1: CSV → MongoDB
│   ├── data_validation.py             # Stage 2: Quality checks + Evidently reports
│   ├── data_preprocess.py             # Stage 3: Clean + encode + split data
│   ├── data_train.py                  # Stage 4: Train 3 models + MLflow logging
│   ├── predict.py                     # Prediction helper used by API
│   ├── monitor.py                     # Production drift detection
│   └── retrain.py                     # Auto-retraining script
│
├── data/
│   ├── raw/
│   │   ├── dataset_phishing.csv       # Original dataset
│   │   └── dataset_phishing.csv.dvc   # DVC pointer (committed to Git)
│   └── processed/                     # Output of preprocessing stage
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── best_model.pkl                 # Champion model loaded by API
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── logistic_regression.pkl
│
├── reports/
│   ├── validation/                    # Evidently validation HTML reports
│   ├── model_eval/                    # Per-model classification reports
│   └── monitoring/                    # Production monitoring reports
│       ├── drift_report_YYYYMMDD.html # Evidently drift HTML report
│       ├── drift_status.json          # Latest drift check result
│       └── retrain_history.json       # Log of all auto-retrain events
│
├── config/
│   └── best_model.json                # Metrics of the currently deployed model
│
├── .github/
│   └── workflows/
│       ├── ci-cd.yml                  # Main CI/CD: train → Docker Hub → Render
│       └── monitoring.yml             # Nightly: drift detect → retrain → redeploy
│
├── mlruns/                            # MLflow experiment tracking data (auto-created)
│
├── app.py                             # FastAPI application (prediction server)
├── dvc.yaml                           # DVC pipeline stage definitions
├── Dockerfile                         # Container build instructions
├── .dockerignore                      # Files excluded from Docker image
├── .gitignore                         # Files excluded from Git
├── .env                               # Local secrets — NEVER commit this file
├── requirements.txt                   # All Python dependencies
└── README.md                          # This file
```

---

## 12. How to Run This Project

### Prerequisites

- Python 3.11 installed
- Git installed
- Docker Desktop installed (for Docker steps)
- Free MongoDB Atlas account
- Free GitHub account
- Free Docker Hub account
- Free Render.com account

### Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/phishing-mlops.git
cd phishing-mlops
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

**3. Install all dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the project root:
```
MONGO_URI=mongodb+srv://your_username:your_password@cluster0.xxxxx.mongodb.net/
```

**5. Place the dataset**

Download `dataset_phishing.csv` from Kaggle and place at:
```
data/raw/dataset_phishing.csv
```

**6. Run the complete pipeline**
```bash
dvc repro
```

Runs all four stages automatically: ingestion → validation → preprocessing → training.

**7. View MLflow experiment results**
```bash
mlflow ui --workers 1
```
Open `http://localhost:5000` in your browser.

**8. Start the prediction API locally**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/docs` for interactive Swagger documentation.

**9. Run with Docker locally**
```bash
docker build -t phishing-detection .
docker run -p 8000:8000 --env-file .env phishing-detection
```

**10. Run monitoring manually**
```bash
python src/monitor.py
```

**11. Trigger manual retrain**
```bash
python src/retrain.py
```

---

## 13. API Documentation

### Endpoints

#### `GET /`
Home — returns API status and model loaded state.
```json
{
  "message": "Phishing Detection API is running!",
  "status": "healthy",
  "model_loaded": true
}
```

#### `GET /health`
Lightweight health check for monitoring and Docker probes.
```json
{
  "status": "ok",
  "model_ready": true
}
```

#### `GET /model-info`
Returns metrics of the currently deployed model.
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
The main prediction endpoint.

**Request:**
```json
{
  "features": {
    "length_url": 75,
    "nb_dots": 3,
    "nb_hyphens": 2,
    "nb_at": 0,
    "nb_qm": 1,
    "nb_eq": 2,
    "https_token": 1,
    "ip": 0,
    "nb_subdomains": 2,
    "domain_age": 1000,
    "page_rank": 5
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

## 14. Model Results

Three models were trained and compared. Best model selected automatically by F1 score.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** ⭐ | ~97.1% | ~96.9% | ~97.3% | ~97.1% |
| Gradient Boosting | ~96.4% | ~96.1% | ~96.7% | ~96.4% |
| Logistic Regression | ~92.1% | ~91.8% | ~92.4% | ~92.1% |

**Why F1 Score is the primary metric:**
Accuracy alone is misleading for security problems. A model that rarely flags safe sites (high precision) but misses many actual phishing sites (low recall) is dangerous. F1 score is the harmonic mean of both, so it only scores high when both are strong. For phishing detection, missing a real phishing site (low recall) is more costly than occasional false alarms.

---

## 15. CI/CD Pipeline

### Main Pipeline — Runs on Every Push to `main`

```
git push origin main
        │
        ▼
Job 1: Run ML Pipeline (ubuntu-latest)
   Install Python 3.11
   pip install -r requirements.txt
   Create .env from GitHub Secrets
   python src/data_ingestion.py
   python src/data_validation.py
   python src/data_preprocess.py
   python src/data_train.py
   uvicorn app:app & curl /health  ← smoke test
   Upload trained-model artifact
        │
        ▼ (only on push to main, not PRs)
Job 2: Build and Push to Docker Hub
   Download trained-model artifact
   docker login (Docker Hub token)
   docker buildx build + push
   Tagged: latest + commit SHA
        │
        ▼
Job 3: Deploy to Render.com
   curl RENDER_DEPLOY_HOOK
   Wait 90 seconds
   curl RENDER_APP_URL/health  ← verify live
        │
        ▼
🚀 Live at https://your-app.onrender.com
```

### Monitoring Pipeline — Runs Every Night at 2 AM UTC

```
Cron: "0 2 * * *"
        │
        ▼
Job 1: Drift Detection
   Run full data pipeline
   python src/monitor.py
   Upload monitoring report artifact
        │
   ┌────┴──────────────┐
No drift           Drift > 30%
   │                   │
Sleep           Job 2: Auto Retrain
                python src/retrain.py
                        │
                        ▼
                Job 3: Redeploy
                Build new Docker image
                Push to Docker Hub
                Trigger Render deploy
                        │
                        ▼
                New model live automatically ✅
```

### All GitHub Secrets Required

| Secret | Used By |
|--------|---------|
| `MONGO_URI` | All Python scripts |
| `DOCKERHUB_USERNAME` | Docker Hub login |
| `DOCKERHUB_TOKEN` | Docker Hub login |
| `RENDER_DEPLOY_HOOK` | Trigger Render redeploy |
| `RENDER_APP_URL` | Health check after deploy |

---

## 16. Free Tools Used — Zero Cost Stack

This entire project runs on free services. No credit card charges, no surprise bills.

| Tool | Free Plan Details | Our Usage |
|------|------------------|-----------|
| **MongoDB Atlas** | 512 MB storage, free forever | Raw data store |
| **GitHub** | Free for public repos, 2000 CI/CD minutes/month | Code hosting + CI/CD + cron scheduler |
| **Docker Hub** | 1 free public repository | Docker image registry |
| **Render.com** | Free web service, 750 hours/month | Live API hosting |
| **MLflow** | Open source, runs locally | Experiment tracking |
| **Evidently AI** | Open source | Data validation + drift detection |
| **DVC** | Open source | Data versioning + pipeline |

---

## 17. What You Learn From This Project

**Data Engineering**
- Storing and retrieving data from a cloud NoSQL database
- Validating data quality programmatically before training
- Generating professional HTML data quality reports

**Machine Learning Engineering**
- Building modular, reusable preprocessing pipelines
- Training and comparing multiple model types systematically
- Understanding why F1 score matters more than accuracy

**MLOps Fundamentals**
- Versioning data files alongside code using DVC
- Defining reproducible pipelines with dependency tracking
- Tracking experiments so you can always answer which model is in production and why
- Detecting data drift in production using statistical tests
- Building self-healing systems that retrain automatically

**Software Engineering for ML**
- Structuring ML projects as proper Python packages
- Secrets management with environment variables
- Building REST APIs that serve ML predictions
- Containerising ML applications with Docker

**DevOps for ML**
- CI/CD pipelines for ML systems (not just web apps)
- Docker image registries and container deployment
- Scheduled automation workflows (cron jobs)
- Multi-job GitHub Actions workflows with artifact passing between jobs

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
evidently           # Data validation and production drift monitoring
fastapi             # REST API framework
uvicorn             # ASGI server to run FastAPI
joblib              # Model serialisation and deserialisation
```

---

## License

This project is open source and available under the MIT License.

---

## Author

Built as a complete MLOps learning project demonstrating production-grade machine learning engineering practices using entirely free-tier services.

**Live API:** https://phishing-detection-latest.onrender.com
**Docker Image:** https://hub.docker.com/r/sazna/phishing-detection

If you found this helpful, please ⭐ star the repository.
