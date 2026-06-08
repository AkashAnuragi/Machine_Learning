# 🤖 Machine Learning — Complete Learning Repository

> **Module 6: Mastering Machine Learning**
> A structured, beginner-to-advanced journey through core ML concepts, Time Series Analysis, EDA, and NLP.

---

## 👨‍💻 About This Repo

This repository documents my complete hands-on learning of **Machine Learning** — from theory to implementation. Every topic is covered with notes, code, and examples, organized module by module.

---

## 📚 Table of Contents

- [Introduction to Machine Learning](#-introduction-to-machine-learning)
- [Time Series Analysis](#-time-series-analysis)
- [Statistical Foundations & EDA](#-statistical-foundations--eda)
- [Text Mining & NLP](#-text-mining--nlp)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Progress Tracker](#-progress-tracker)
- [Connect With Me](#-connect-with-me)

---

## 🧠 Introduction to Machine Learning

| Topic | Description |
|---|---|
| **What is Machine Learning?** | Definition, how computers learn from data |
| **ML Use-Cases** | Real-world applications across industries |
| **ML Process Flow** | Data → Model → Prediction pipeline |
| **ML Categories** | Supervised, Unsupervised, Reinforcement Learning |

### ML Categories at a Glance

```
Machine Learning
├── Supervised Learning      (labeled data → predict output)
│   ├── Regression           (continuous values: price, salary)
│   └── Classification       (categories: spam/not spam, yes/no)
│
├── Unsupervised Learning    (unlabeled data → find patterns)
│   ├── Clustering           (KMeans, DBSCAN)
│   ├── Dimensionality Reduction  (PCA, t-SNE, UMAP, LDA)
│   └── Association Rules    (Apriori, FP-Growth)
│
└── Reinforcement Learning   (no dataset → reward & punishment)
    └── Trial & Error         (games, autopilot, robotics)
```

---

## 📈 Time Series Analysis

> Analyzing data points collected over time to identify trends, patterns, and make future predictions.

| Topic | Description |
|---|---|
| **What is TSA?** | Introduction to time-indexed data |
| **Importance of TSA** | Why time context matters in predictions |
| **Components of TSA** | Trend, Seasonality, Cyclicity, Irregularity |
| **White Noise** | Random variation with no pattern |
| **AR Model** | AutoRegressive — depends on past values |
| **MA Model** | Moving Average — depends on past errors |
| **ARMA Model** | Combination of AR + MA |
| **ARIMA Model** | ARMA + Differencing for non-stationary data |
| **Stationarity** | Making a time series stationary |
| **ACF & PACF** | AutoCorrelation & Partial AutoCorrelation Functions |

### Key Formula — ARIMA

```
ARIMA(p, d, q)
 p = number of AR terms
 d = number of differences required
 q = number of MA terms
```

---

## 📊 Statistical Foundations & EDA

> Exploratory Data Analysis — understanding data before modeling.

### EDA Techniques

```
EDA
├── Univariate Analysis      (one variable at a time)
│   ├── Non-Graphical        (mean, median, mode, std, variance)
│   └── Graphical            (histogram, box plot, KDE)
│
└── Multivariate Analysis    (two or more variables)
    ├── Non-Graphical        (correlation matrix, cross-tabulation)
    └── Graphical            (scatter plot, pair plot, heat map)
```

| EDA Type | Methods Used |
|---|---|
| Univariate Non-Graphical | Summary stats, frequency tables |
| Univariate Graphical | Histogram, Box plot, Density plot |
| Multivariate Non-Graphical | Correlation matrix, Covariance |
| Multivariate Graphical | Scatter plot, Pair plot, Heatmap |
| **Heat Maps** | Visualizing correlation between all features |

---

## 🔤 Text Mining & NLP

> Extracting meaningful information from unstructured text data.

| Topic | Description |
|---|---|
| **Overview of Text Mining** | What is text mining and why it matters |
| **Need of Text Mining** | Business value of unstructured data |
| **NLP in Text Mining** | How NLP powers text understanding |

### NLP Pipeline (Overview)

```
Raw Text
   ↓
Tokenization
   ↓
Stop Word Removal
   ↓
Stemming / Lemmatization
   ↓
Feature Extraction (TF-IDF / Bag of Words)
   ↓
Model / Analysis
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 📁 Folder Structure

```
Machine_Learning/
│
├── 01_Introduction/
│   ├── what_is_ml.ipynb
│   ├── ml_usecases.ipynb
│   ├── ml_process_flow.ipynb
│   └── ml_categories.ipynb
│
├── 02_Time_Series_Analysis/
│   ├── intro_to_tsa.ipynb
│   ├── components_of_tsa.ipynb
│   ├── white_noise.ipynb
│   ├── AR_MA_ARMA_models.ipynb
│   ├── ARIMA_model.ipynb
│   ├── stationarity.ipynb
│   └── ACF_PACF.ipynb
│
├── 03_EDA_Statistical_Foundations/
│   ├── what_is_eda.ipynb
│   ├── univariate_non_graphical.ipynb
│   ├── univariate_graphical.ipynb
│   ├── multivariate_non_graphical.ipynb
│   ├── multivariate_graphical.ipynb
│   └── heat_maps.ipynb
│
├── 04_Text_Mining_NLP/
│   ├── overview_text_mining.ipynb
│   ├── need_of_text_mining.ipynb
│   └── nlp_in_text_mining.ipynb
│
├── datasets/
│   └── (datasets used across notebooks)
│
├── notes/
│   └── (handwritten notes, PDFs, references)
│
└── README.md
```

---

## ✅ Progress Tracker

### Module 6 — Mastering Machine Learning

| # | Topic | Status |
|---|---|---|
| 1 | What is Machine Learning? | ✅ Done |
| 2 | ML Use-Cases | ✅ Done |
| 3 | ML Process Flow | ✅ Done |
| 4 | ML Categories | ✅ Done |
| 5 | What is Time Series Analysis? | ✅ Done |
| 6 | Importance of TSA | ✅ Done |
| 7 | Components of TSA | ✅ Done |
| 8 | White Noise | ✅ Done |
| 9 | AR Model | ✅ Done |
| 10 | MA Model | ✅ Done |
| 11 | ARMA Model | ✅ Done |
| 12 | ARIMA Model | ✅ Done |
| 13 | Stationarity | ✅ Done |
| 14 | ACF & PACF | ✅ Done |
| 15 | What is EDA? | ✅ Done |
| 16 | EDA Techniques & Classification | ✅ Done |
| 17 | Univariate Non-Graphical EDA | ✅ Done |
| 18 | Univariate Graphical EDA | ✅ Done |
| 19 | Multivariate Non-Graphical EDA | ✅ Done |
| 20 | Multivariate Graphical EDA | ✅ Done |
| 21 | Heat Maps | ✅ Done |
| 22 | Overview of Text Mining | ✅ Done |
| 23 | Need of Text Mining | ✅ Done |
| 24 | NLP in Text Mining | ✅ Done |

---

## 🔗 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-akash--anuragi-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/akash-anuragi)
[![GitHub](https://img.shields.io/badge/GitHub-AkashAnuragi-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AkashAnuragi)
[![LeetCode](https://img.shields.io/badge/LeetCode-AkashAnuragi-FFA116?style=for-the-badge&logo=leetcode&logoColor=white)](https://leetcode.com/AkashAnuragi)

---

<div align="center">

**⭐ If this repo helped you, please give it a star!**

*Made with 💙 while learning Machine Learning — one concept at a time.*

</div>
