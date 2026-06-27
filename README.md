# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations in E-Commerce

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-lightblue?logo=plotly)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Task-Unsupervised%20Learning-green)

An end-to-end AI/ML pipeline that transforms raw e-commerce transaction data into actionable business intelligence. By analyzing purchasing behaviors, segmenting customers via RFM analysis, and deploying a collaborative filtering recommendation engine, this project provides a comprehensive toolkit for targeted marketing, customer retention, and cross-selling.

---

## 📑 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [📌 Problem Statement](#-problem-statement)
- [🛠️ Tech Stack](#-tech-stack)
- [📊 Dataset Description](#-dataset-description)
- [🚀 Project Pipeline](#-project-pipeline)
- [📈 Key EDA Insights](#-key-eda-insights)
- [🧠 Clustering & Model Evaluation](#-clustering--model-evaluation)
- [🤝 Recommendation System](#-recommendation-system)
- [🔬 Statistical Hypothesis Testing](#-statistical-hypothesis-testing)
- [📱 Streamlit Web Application](#-streamlit-web-application)
- [⚙️ Installation & Usage](#️-installation--usage)

---

## 🎯 Project Overview
The global e-commerce industry generates vast amounts of transaction data daily. **Shopper Spectrum** leverages this data to uncover hidden patterns in customer purchasing behavior. 

The project is divided into two core deliverables:
1. **Customer Segmentation:** Using RFM (Recency, Frequency, Monetary) analysis combined with K-Means, Hierarchical, and DBSCAN clustering to categorize customers into actionable tiers (High-Value, Regular, Occasional, At-Risk).
2. **Product Recommendations:** An Item-Based Collaborative Filtering system that suggests the top 5 most similar products based on co-purchase history, driving cross-sales and increasing Average Order Value (AOV).

---

## 📌 Problem Statement
Online retailers struggle to understand the diverse purchasing behaviors of their customer base, leading to generic, ineffective marketing strategies. Specifically, the business faces three core challenges:
1. **Lack of Customer Differentiation:** No data-driven method to identify high-value loyalists vs. churning at-risk customers.
2. **Low Cross-Selling Capability:** The platform lacks personalized product recommendations.
3. **Data Quality & Complexity:** The raw transactional dataset contains anomalies (cancelled orders, missing IDs, negative values) that obscure true purchasing trends.

---

## 🛠️ Tech Stack
| Category | Tools/Libraries |
| :--- | :--- |
| **Language** | Python |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, SciPy |
| **Data Visualization** | Plotly, Matplotlib |
| **Web Framework** | Streamlit |
| **Model Serialization** | Joblib, Pickle |

---

## 📊 Dataset Description
The dataset contains transactional data for a UK-based online retailer from 2022–2023.
- **Size:** ~540,000 transactions
- **Features:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

---

## 🚀 Project Pipeline

### 1. Data Preprocessing & Feature Engineering
*   Dropped rows with missing `CustomerID`.
*   Excluded cancelled invoices (`InvoiceNo` starting with 'C').
*   Removed negative/zero `Quantity` and `UnitPrice`.
*   Engineered features: `TotalAmount`, `InvoiceMonth`, `DayOfWeek`, `Hour`.

### 2. Exploratory Data Analysis (EDA)
*   Generated **20 interactive Plotly charts** (Univariate, Bivariate, Multivariate) to uncover trends in geography, temporal seasonality, product popularity, and customer spend distributions.

### 3. RFM Feature Engineering
*   **Recency:** Days since the customer's last purchase.
*   **Frequency:** Total number of unique transactions.
*   **Monetary:** Total spend (`Quantity * UnitPrice`).
*   Applied `log1p` transformation and `StandardScaler` to handle extreme skewness in the RFM data.

### 4. Clustering Methodology
Implemented three clustering algorithms to validate groupings and detect anomalies:
*   **K-Means:** Primary model. Used Elbow Curve & Silhouette Score to determine optimal $k=4$.
*   **Hierarchical Clustering:** Used a Dendrogram (Ward's method) to validate the nested sub-structures of the 4 clusters.
*   **DBSCAN:** Used a K-Distance graph to find `eps`. Deployed specifically for anomaly/noise detection to flag bulk buyers/fraud.

### 5. Recommendation System
*   Built a Customer-Product matrix using `pivot_table`.
*   Computed **Cosine Similarity** between item columns.
*   Created a function to return the Top 5 most similar products for any given product name.

---

## 📈 Key EDA Insights
1. **Geographic Concentration:** The UK accounts for >90% of transactions, representing a geographic concentration risk.
2. **Seasonality:** Massive transaction spikes from September to November, requiring front-loaded Q3/Q4 inventory planning.
3. **Temporal Behavior:** Peak shopping hours are between 10:00 AM and 3:00 PM.
4. **Pareto Principle:** The top 15 customers drive a disproportionately large percentage of total revenue.

---

## 🧠 Clustering & Model Evaluation

### The 4 Customer Segments (K-Means)
| Segment | Recency | Frequency | Monetary | Business Action |
| :--- | :--- | :--- | :--- | :--- |
| **High-Value** | Low | High | High | VIP loyalty perks, early access to products. |
| **Regular** | Medium | Medium | Medium | Upsell premium products, cross-sell accessories. |
| **Occasional** | High | Low | Low | Re-engagement discounts, targeted emails. |
| **At-Risk** | Very High | Medium | Medium | Urgent win-back campaigns before churn. |

### Model Comparison
| Algorithm | Silhouette Score (Higher is better) | Business Use Case |
| :--- | :--- | :--- |
| **K-Means** | ~0.42 (Best) | Primary segmentation for marketing tiers. |
| **Hierarchical** | ~0.40 | Validated nested sub-groups (B2B vs Retail VIPs). |
| **DBSCAN** | ~0.28 | Anomaly detection (flagged extreme bulk buyers as Noise). |

---

## 🤝 Recommendation System
The Item-Based Collaborative Filtering engine maps product affinities. For example, entering *"WHITE HANGING HEART T-LIGHT HOLDER"* returns:
1. WHITE METAL LANTERN
2. CREAM CUPID HEARTS COAT HANGER
3. RED WOOLLY HOTTIE WHITE HEART
4. GLASS STAR FROSTED T-LIGHT HOLDER
5. SET 7 BABUSHKA NESTING BOXES

---

## 🔬 Statistical Hypothesis Testing
Three tests were conducted to mathematically validate business assumptions:
1. **ANOVA:** Proved that Average Order Value (AOV) significantly differs across the Top 5 countries (p < 0.05).
2. **Pearson Correlation:** Proved a significant negative correlation between Recency and Monetary value (recent customers spend more).
3. **Two-Sample T-Test:** Proved that the purchase frequency of High-Value vs. At-Risk customers is statistically different, validating the K-Means clusters.

---

## 📱 Streamlit Web Application
The entire pipeline is encapsulated in an interactive Streamlit web app with two modules:
1. **Product Recommendation Module:** Text input for a product name → returns 5 similar products with similarity scores.
2. **Customer Segmentation Module:** Number inputs for Recency, Frequency, and Monetary → predicts the customer's cluster segment and recommends a business action.

To run the app:
```bash
streamlit run app.py
```

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/shopper-spectrum.git
cd shopper-spectrum
```

2. **Create and activate a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Jupyter Notebook (for code review):**
```bash
jupyter notebook Shopper_Spectrum.ipynb
```

5. **Launch the Streamlit App:**
```bash
streamlit run app.py
```

---
*Developed as part of an AI/ML Specialist pipeline for E-Commerce and Retail Analytics.*
