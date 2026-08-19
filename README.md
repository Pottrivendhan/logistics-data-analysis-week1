# 🚚 Logistics Data Analysis – Week 1 to Week 4

A complete Python-based logistics analytics project covering **strategic planning, data preprocessing, exploratory data analysis, predictive modeling, and logistics optimization**.

---

## 📌 Project Overview

This project demonstrates an end-to-end data science workflow for analyzing logistics and e-commerce delivery operations.

The project is divided into four progressive stages:

| Week | Focus | Main Outcome |
|------|-------|--------------|
| **Week 1** | Strategic Planning & Data Exploration | Logistics KPIs and business insights |
| **Week 2** | Data Cleaning & Preprocessing | Clean and analysis-ready datasets |
| **Week 3** | Advanced EDA & Visualization | 12 logistics performance visualizations |
| **Week 4** | Predictive Modeling & Optimization | Delivery-time prediction and risk prioritization |

The project uses the **Olist Brazilian E-Commerce dataset** and Python data science technologies.

---

# 🎯 Project Objectives

The main objectives are to:

- Analyze logistics delivery performance.
- Calculate important logistics KPIs.
- Identify delivery delays and high-risk regions.
- Clean and preprocess logistics datasets.
- Explore trends and relationships using visualization.
- Build predictive models for delivery-time forecasting.
- Compare machine learning models.
- Tune the best-performing model.
- Classify shipments according to predicted delivery risk.
- Develop an optimization strategy for logistics operations.
- Support data-driven logistics decision-making.

---

# 🛠️ Technologies Used

- **Python 3.12**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Jupyter / Python scripts**
- **Git**
- **GitHub**

---

# 📂 Dataset

The project uses the **Olist Brazilian E-Commerce dataset**.

Major datasets include:

- Orders
- Order Items
- Customers
- Sellers
- Products

Important attributes include:

- Order ID
- Customer ID
- Order Status
- Purchase Timestamp
- Delivery Timestamp
- Estimated Delivery Date
- Product Price
- Freight Value
- Seller ID
- Product ID
- Customer State
- Product Weight
- Product Dimensions

The original datasets are excluded from GitHub using `.gitignore` to avoid unnecessarily large repository files.

---

# 📅 Week 1 – Strategic Planning and Data Exploration

## Objective

The first week focused on defining the logistics problem, identifying KPIs, exploring the dataset, and establishing a strategic analytical roadmap.

## Activities

- Loaded the logistics dataset.
- Examined dataset structure.
- Identified missing values.
- Calculated logistics KPIs.
- Analyzed delivery performance.
- Examined seller performance.
- Analyzed state-level delivery performance.
- Investigated freight costs.
- Performed seller clustering.

## Key KPIs

| KPI | Result |
|---|---:|
| Total Orders | 99,441 |
| Delivered Orders | 96,476 |
| Average Delivery Time | 12.56 days |
| Late Delivery Rate | 8.11% |
| On-Time Delivery Rate | 91.89% |
| Average Freight Cost | 22.82 |

## Machine Learning

A classification model was developed to identify delivery outcomes.

Model accuracy:

**91.42%**

The analysis also used **K-Means clustering** to segment sellers based on operational behavior.

### Seller Clusters

| Cluster | Sellers | Avg Orders | Avg Delivery Days | Avg Late Rate |
|---|---:|---:|---:|---:|
| 0 | 939 | 51.31 | 16.10 | 14.58% |
| 1 | 91 | 2.16 | 33.54 | 82.77% |
| 2 | 1,921 | 15.06 | 9.30 | 1.89% |
| 3 | 19 | 1,078.84 | 13.04 | 8.32% |

---

# 📅 Week 2 – Data Collection, Cleaning and Preprocessing

## Objective

Week 2 focused on preparing high-quality data for analysis and machine learning.

## Data Quality Analysis

The preprocessing pipeline examined:

- Data types
- Missing values
- Duplicate records
- Unique values
- Numerical distributions
- Product characteristics
- Outliers

### Missing Values

Important missing values were identified in:

- `order_approved_at`
- `order_delivered_carrier_date`
- `order_delivered_customer_date`

Missing percentages were calculated to understand their impact.

## Duplicate Analysis

The dataset was checked for:

- Duplicate orders
- Duplicate order IDs

No duplicate order IDs were identified in the analysis.

## Numerical Analysis

Important variables analyzed included:

- Product price
- Freight value
- Product weight
- Product length
- Product height
- Product width

## Preprocessing Techniques

The project applied:

```text
Data Collection
      ↓
Data Inspection
      ↓
Missing Value Analysis
      ↓
Duplicate Detection
      ↓
Outlier Analysis
      ↓
Data Transformation
      ↓
Normalization / Preparation
      ↓
Clean Dataset
