🚚 Logistics Data Analysis — Week 1 to Week 4

A complete end-to-end Logistics Data Analytics and Machine Learning project developed using Python. This project progresses from logistics KPI analysis and data preprocessing to exploratory visualization, predictive modeling, risk classification, and operational optimization.

📌 Project Overview

This project analyzes logistics and e-commerce delivery data to understand operational performance, identify delivery delays, discover high-risk sellers and regions, and build a predictive system for delivery-time forecasting.

The project is divided into four stages:

Week 1: Logistics KPI Analysis and Strategic Insights
Week 2: Data Collection, Cleaning and Preprocessing
Week 3: Advanced Exploratory Data Analysis and Visualization
Week 4: Predictive Modeling and Logistics Optimization

The project uses the Olist Brazilian E-Commerce dataset and Python-based data science and machine learning techniques.

🎯 Objectives

The main objectives of this project are:

Analyze logistics delivery performance.
Calculate important logistics KPIs.
Identify late and on-time deliveries.
Analyze seller and regional performance.
Detect data-quality problems.
Clean and preprocess logistics datasets.
Explore trends and relationships through visualization.
Build machine learning models for delivery-time prediction.
Compare different predictive models.
Tune the best-performing model.
Classify shipments according to predicted delivery risk.
Prioritize high-risk shipments.
Develop practical logistics optimization strategies.
Support data-driven operational decision-making.
🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
Pandas	Data manipulation and analysis
NumPy	Numerical computation
Matplotlib	Data visualization
Scikit-learn	Machine learning and preprocessing
Git	Version control
GitHub	Project repository
PowerShell	Project execution
📊 Dataset

The project uses the Olist Brazilian E-Commerce dataset.

The major datasets used include:

olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_customers_dataset.csv
olist_sellers_dataset.csv
olist_products_dataset.csv
Important Variables

The analysis uses information such as:

Order ID
Customer ID
Seller ID
Product ID
Order status
Order purchase timestamp
Order approval timestamp
Carrier delivery date
Customer delivery date
Estimated delivery date
Product price
Freight value
Customer state
Product weight
Product dimensions
Number of items

Original raw datasets are excluded from GitHub where appropriate using .gitignore to prevent unnecessary repository size.

📅 WEEK 1 — Logistics KPI Analysis
Objective

Week 1 focused on understanding the logistics dataset and establishing important operational performance indicators.

The analysis examined order volume, delivery time, late-delivery performance, freight costs, seller performance, state-level performance, and product/freight characteristics.

Week 1 Workflow
Raw Logistics Data
        ↓
Data Loading
        ↓
Data Exploration
        ↓
KPI Calculation
        ↓
Seller Analysis
        ↓
State Analysis
        ↓
Product & Freight Analysis
        ↓
Machine Learning
        ↓
Seller Clustering
Key Logistics KPIs
KPI	Result
Total Orders	99,441
Delivered Orders	96,476
Average Delivery Time	12.56 days
Late Delivery Rate	8.11%
On-Time Delivery Rate	91.89%
Average Freight Cost	22.82

These KPIs provide a baseline understanding of overall logistics performance.

Monthly Analysis

The monthly analysis examined:

Order volume
Average delivery time
Late-delivery rate
Delivery performance over time

A significant increase in late deliveries was observed during some periods, including:

March 2018 — 21.36% late rate

This indicates the importance of monitoring seasonal or operational changes.

Seller Analysis

The project identified:

Top sellers by order volume.
High-risk sellers.
Seller delivery performance.
Seller late-delivery rates.

High-risk sellers were identified based on their late-delivery performance.

State-Level Analysis

Delivery performance was analyzed across Brazilian customer states.

Some high-risk states included:

State	Late Rate
AL	23.93%
MA	19.67%
PI	15.97%
CE	15.32%
SE	15.22%
BA	14.04%
RJ	13.47%

This analysis helps identify regions requiring additional logistics attention.

🤖 Week 1 Machine Learning

A classification model was developed to analyze delivery outcomes.

Model Accuracy

91.42%

Classification Results
              precision    recall  f1-score


Class 0          0.92       0.99      0.95
Class 1          0.32       0.05      0.09

The results also demonstrate that accuracy alone is not sufficient for evaluating an imbalanced logistics classification problem.

👥 Seller Clustering

K-Means clustering was used to segment sellers based on:

Order volume
Average delivery time
Late-delivery rate
Cluster Summary
Cluster	Sellers	Avg Orders	Avg Delivery Days	Avg Late Rate
0	939	51.31	16.10	14.58%
1	91	2.16	33.54	82.77%
2	1,921	15.06	9.30	1.89%
3	19	1,078.84	13.04	8.32%
Key Insight

Cluster 1 represents a small group of sellers with:

Very low order volume.
Very high average delivery time.
Very high late-delivery rate.

This group can be considered a high-priority segment for further investigation.

📅 WEEK 2 — Data Collection, Cleaning and Preprocessing
Objective

Week 2 focused on preparing reliable and analysis-ready datasets.

The major goal was to identify and handle data-quality issues before advanced analysis and machine learning.

🔍 Data Quality Analysis

The following checks were performed:

Data types
Missing values
Missing percentages
Duplicate records
Duplicate order IDs
Unique values
Numerical distributions
Product statistics
Potential outliers
Missing Value Analysis

Important missing values were identified in:

order_approved_at
order_delivered_carrier_date
order_delivered_customer_date

Observed missing percentages included approximately:

Column	Missing %
order_approved_at	0.16%
order_delivered_carrier_date	1.79%
order_delivered_customer_date	2.98%

The impact of missing values was evaluated before preprocessing.

🔁 Duplicate Analysis

The dataset was checked for:

Duplicate orders
Duplicate order IDs

The analysis found:

Duplicate orders: 0
Duplicate order IDs: 0
📐 Numerical Analysis

Important numerical variables included:

Product price
Freight value
Product weight
Product length
Product height
Product width

Example statistics:

Price
Mean:       120.65
Median:      74.99
Maximum:   6735.00
Freight
Mean:        19.99
Maximum:    409.68
Product Weight
Mean:      2276.47 grams
Maximum: 40425.00 grams
🧹 Preprocessing Techniques

The Week 2 pipeline followed:

Data Collection
       ↓
Data Inspection
       ↓
Missing Value Analysis
       ↓
Duplicate Detection
       ↓
Numerical Analysis
       ↓
Outlier Investigation
       ↓
Data Transformation
       ↓
Clean Dataset

The cleaned data was prepared for Week 3 exploratory analysis and Week 4 machine learning.

📅 WEEK 3 — Advanced EDA and Visualization
Objective

Week 3 focused on discovering patterns, relationships, trends, and operational bottlenecks through exploratory data analysis and visualization.

📈 Delivery Performance

The analysis found:

Metric	Result
On-Time Deliveries	88,652
Late Deliveries	7,826
Average On-Time Delivery	10.88 days
Average Late Delivery	31.52 days

The large difference between average on-time and late delivery duration demonstrates the operational impact of delayed shipments.

📊 Week 3 Visualizations

The project generated the following visualizations:

week3_charts/
│
├── correlation_heatmap.png
├── delivery_time_comparison.png
├── delivery_time_distribution.png
├── freight_distribution.png
├── freight_vs_delivery_time.png
├── high_risk_sellers.png
├── monthly_late_rate.png
├── monthly_order_volume.png
├── on_time_vs_late.png
├── price_vs_freight.png
├── state_late_delivery_rate.png
└── weight_vs_freight.png
Visualization Purpose
1. Delivery Time Distribution

Shows how delivery times are distributed across orders.

2. Monthly Order Volume

Shows changes in order volume over time.

3. Monthly Late Rate

Identifies periods with increased delivery delays.

4. Freight Distribution

Shows the distribution of transportation costs.

5. Price vs Freight

Examines the relationship between product price and freight value.

6. Weight vs Freight

Examines how product weight relates to freight cost.

7. State Late Delivery Rate

Identifies high-risk geographical regions.

8. High-Risk Sellers
 
Highlights sellers with poor delivery performance.

9. Correlation Heatmap

Shows relationships between numerical logistics variables.

10. Freight vs Delivery Time

Examines whether freight cost is associated with delivery duration.

11. On-Time vs Late Delivery

Compares delivery performance between successful and delayed shipments.

12. Delivery Time Comparison

Compares delivery duration across delivery categories.

📊 Correlation Analysis

Important correlations observed during Week 3 included:

Variable Pair	Correlation
Delivery Days – Late Delivery	0.586
Delivery Days – Freight	0.215
Delivery Days – Price	0.062
Late Delivery – Freight	0.040
Price – Freight	0.413

The results indicate that delivery status has a stronger relationship with delivery duration than the basic price or freight variables.

📅 WEEK 4 — Predictive Modeling and Optimization
Objective

Week 4 extended the project from descriptive analytics to predictive analytics.

The main objective was:

Predict delivery time and use predictions to prioritize logistics operations.

🎯 Prediction Target

The target variable is:

delivery_days

It represents the number of days between:

Order Purchase
      ↓
Customer Delivery
🧩 Features Used

The predictive model uses:

total_price
total_freight
average_price
average_freight
item_count
customer_state
order_month
order_dayofweek
order_year
📊 Modeling Dataset

The final modeling dataset contained:

96,470 records

The train-test split was:

Dataset	Records	Percentage
Training	77,176	80%
Testing	19,294	20%
⚙️ Preprocessing

The Week 4 pipeline used:

Median imputation for numerical variables.
Most-frequent imputation for categorical variables.
One-hot encoding for customer state.
Scikit-learn Pipeline.
Scikit-learn ColumnTransformer.
🤖 Predictive Models

Two initial models were evaluated:

1. Linear Regression

Used as a baseline model.

2. Random Forest Regressor

Used to capture nonlinear relationships and interactions among logistics variables.

📈 Model Evaluation

The following metrics were used:

MAE

Mean Absolute Error measures the average absolute prediction error.

Lower is better.

RMSE

Root Mean Squared Error penalizes larger prediction errors more heavily.

Lower is better.

R²

R² measures the proportion of variation explained by the model.

Higher is better.

🏆 Model Comparison
Model	MAE	RMSE	R²
Linear Regression	5.4962	8.3312	0.1998
Random Forest	4.9385	7.8805	0.2840
Tuned Random Forest	4.9300	7.7763	0.3029
⭐ Final Model

The final selected model is:

Tuned Random Forest Regressor

Best parameters:

n_estimators = 100
max_depth = 10
min_samples_split = 5

Cross-validation:

2-fold Cross Validation

Best cross-validation RMSE:

8.1482

Final test performance:

MAE  = 4.9300 days
RMSE = 7.7763 days
R²   = 0.3029
🔧 Hyperparameter Tuning

GridSearchCV was used to test:

param_grid = {
    "n_estimators": [100],
    "max_depth": [10, 15],
    "min_samples_split": [2, 5]
}

The final configuration selected:

n_estimators = 100
max_depth = 10
min_samples_split = 5

Tuning improved the original Random Forest:

Original Random Forest
RMSE = 7.8805
R²   = 0.2840


Tuned Random Forest
RMSE = 7.7763
R²   = 0.3029
🚨 Delivery Risk Classification

Predicted delivery duration was converted into operational risk categories.

Predicted Delivery Days
          │
          ├── ≤ 10 days
          │      ↓
          │   LOW RISK
          │
          ├── > 10 and ≤ 20 days
          │      ↓
          │  MEDIUM RISK
          │
          └── > 20 days
                 ↓
             HIGH RISK
📊 Risk Distribution
Risk Level	Orders	Percentage	Recommended Action
Low Risk	6,925	35.89%	Normal processing
Medium Risk	10,140	52.56%	Monitor shipment
High Risk	2,229	11.55%	Immediate intervention

The system identified 2,229 high-risk orders in the test set.

⚡ Optimization Layer

The predictive model was converted into an operational decision-support system.

High Risk
Action: Immediate Intervention

Recommended activities:

Investigate shipment status.
Check logistics bottlenecks.
Review route/carrier information.
Prioritize operational support.
Consider proactive customer communication.
Medium Risk
Action: Monitor Shipment

Recommended activities:

Track shipment progress.
Monitor delivery status.
Escalate if the predicted risk increases.
Low Risk
Action: Normal Processing

Recommended activities:

Continue normal logistics workflow.
Avoid unnecessary resource allocation.
🎯 Priority Scoring

A priority score was calculated based on predicted delivery duration.

The highest predicted delivery durations included:

73.96 days
66.40 days
48.87 days
45.80 days
41.33 days
40.36 days
39.96 days
39.78 days
39.51 days
39.37 days

These orders represent the highest predicted delivery-time cases and can receive immediate attention.

📌 Optimization Results

The final optimization layer generated:

week4_optimization_summary.csv
week4_priority_orders.csv
week4_final_predictions.csv

Recommended actions:

High Risk
    ↓
Immediate Intervention


Medium Risk
    ↓
Monitor Shipment


Low Risk
    ↓
Normal Processing
📉 Prediction Error Analysis

Average prediction error by risk category:

Risk Level	Average Prediction Error
Low Risk	3.17 days
Medium Risk	5.22 days
High Risk	9.10 days

This demonstrates an important limitation: high-risk predictions have a larger average error.

Therefore, the risk system should be treated as an early-warning and prioritization tool, not as a guarantee that an individual shipment will be late.

💡 Business Recommendations
1. Predictive Early-Warning System

Use the final model to identify potentially delayed shipments before they become critical.

2. High-Risk Shipment Prioritization

Create a daily operational queue containing high-risk shipments.

3. Regional Logistics Improvement

Investigate states with consistently high late-delivery rates.

4. Seller Performance Monitoring

Monitor sellers with high late-delivery rates and investigate recurring operational problems.

5. Peak-Period Planning

Use historical monthly trends to prepare additional logistics capacity during high-risk periods.

6. Freight Optimization

Analyze high-freight shipments for:

Packaging improvements.
Shipment consolidation.
Carrier selection.
Weight optimization.
7. Model Improvement

Future versions should include additional features such as:

Carrier information.
Route distance.
Seller location.
Product dimensions.
Historical seller performance.
Transportation method.
Regional delivery history.
⚠️ Limitations

The current system has several limitations:

The final R² is 0.3029, meaning the current feature set explains approximately 30.29% of delivery-time variation.
Important operational features such as carrier and route distance are not included.
The risk thresholds of 10 and 20 days are rule-based.
High-risk predictions have an average error of 9.10 days.
The model should therefore support human decision-making rather than replace operational judgment.
Random train-test splitting was used; future versions could use time-based validation for more realistic forecasting.
🔮 Future Enhancements

Future versions of this project can include:

Real-time shipment tracking.
Carrier performance analysis.
Route-distance features.
GPS-based logistics monitoring.
XGBoost or Gradient Boosting models.
Time-series forecasting.
Route optimization.
Automated delay alerts.
Power BI dashboard.
Streamlit dashboard.
FastAPI model deployment.
Real-time logistics prediction API.
Automated daily high-risk shipment reports.
🔄 Complete Project Workflow
                    RAW LOGISTICS DATA
                           │
                           ↓
                  ┌─────────────────┐
                  │     WEEK 1      │
                  │ KPI Analysis    │
                  │ Seller Analysis │
                  │ State Analysis  │
                  │ Clustering      │
                  └────────┬────────┘
                           │
                           ↓
                  ┌─────────────────┐
                  │     WEEK 2      │
                  │ Data Cleaning   │
                  │ Missing Values  │
                  │ Duplicates      │
                  │ Preprocessing   │
                  └────────┬────────┘
                           │
                           ↓
                  ┌─────────────────┐
                  │     WEEK 3      │
                  │ EDA             │
                  │ Visualization   │
                  │ Correlation     │
                  │ Insights        │
                  └────────┬────────┘
                           │
                           ↓
                  ┌─────────────────┐
                  │     WEEK 4      │
                  │ ML Prediction   │
                  │ Model Tuning    │
                  │ Risk Detection  │
                  │ Optimization    │
                  └────────┬────────┘
                           │
                           ↓
                DATA-DRIVEN LOGISTICS
                    DECISION MAKING
📁 Repository Structure
Logistics_Task1/
│
├── data/
│
├── week3_charts/
│   ├── correlation_heatmap.png
│   ├── delivery_time_comparison.png
│   ├── delivery_time_distribution.png
│   ├── freight_distribution.png
│   ├── freight_vs_delivery_time.png
│   ├── high_risk_sellers.png
│   ├── monthly_late_rate.png
│   ├── monthly_order_volume.png
│   ├── on_time_vs_late.png
│   ├── price_vs_freight.png
│   ├── state_late_delivery_rate.png
│   └── weight_vs_freight.png
│
├── logistics_analysis.py
├── week2_preprocessing.py
├── week3_eda_visualization.py
├── week4_predictive_optimization.py
│
├── week4_delivery_predictions.csv
├── week4_final_model_comparison.csv
├── week4_final_predictions.csv
├── week4_high_risk_orders.csv
├── week4_model_comparison.csv
├── week4_optimization_summary.csv
└── week4_priority_orders.csv
📄 Week-by-Week Deliverables
Week 1
logistics_analysis.py

Includes:

KPI analysis
Monthly analysis
Seller analysis
State analysis
Product/freight analysis
Classification
Seller clustering
Week 2
week2_preprocessing.py

Includes:

Data inspection
Missing-value analysis
Duplicate analysis
Numerical analysis
Data cleaning
Preprocessing
Week 3
week3_eda_visualization.py

Includes:

Exploratory data analysis
Statistical analysis
12 visualization outputs
Correlation analysis
Regional analysis
Seller analysis
Week 4
week4_predictive_optimization.py

Includes:

Feature engineering
Preprocessing
Train/test split
Linear Regression
Random Forest
Hyperparameter tuning
Cross-validation
Delivery-time prediction
Risk classification
Priority scoring
Optimization recommendations
▶️ How to Run the Project
Clone the Repository
git clone https://github.com/Pottrivendhan/logistics-data-analysis-week1.git

Navigate to the project:

cd logistics-data-analysis-week1
Install Required Libraries
pip install pandas numpy matplotlib scikit-learn
Run Week 1
python logistics_analysis.py
Run Week 2
python week2_preprocessing.py
Run Week 3
python week3_eda_visualization.py

Charts will be generated in:

week3_charts/
Run Week 4
python week4_predictive_optimization.py

The Week 4 script generates prediction, model-comparison, risk, and optimization CSV files.

📊 Final Project Results
Overall Logistics Performance
Total Orders       : 99,441
Delivered Orders   : 96,476
Avg Delivery Time  : 12.56 days
Late Rate          : 8.11%
On-Time Rate       : 91.89%
Final Machine Learning Model
Model              : Tuned Random Forest
MAE                : 4.9300 days
RMSE               : 7.7763 days
R²                 : 0.3029
Risk Detection
Low Risk           : 6,925
Medium Risk        : 10,140
High Risk          : 2,229
Main Outcome

The project transforms logistics data into a complete analytical and predictive workflow:

DATA
  ↓
CLEANING
  ↓
EXPLORATION
  ↓
VISUALIZATION
  ↓
PREDICTION
  ↓
RISK CLASSIFICATION
  ↓
PRIORITIZATION
  ↓
OPTIMIZATION
🏆 Conclusion

This four-week logistics analytics project demonstrates an end-to-end application of data science to logistics operations.

Week 1 established logistics KPIs and identified important seller and regional performance patterns.

Week 2 focused on data quality, cleaning, missing values, duplicate detection, and preprocessing.

Week 3 transformed the prepared data into meaningful exploratory insights through statistical analysis and 12 visualizations.

Week 4 advanced the project into predictive analytics by developing Linear Regression and Random Forest models. Hyperparameter tuning produced the final Tuned Random Forest model, achieving an MAE of 4.9300 days, RMSE of 7.7763 days, and R² of 0.3029.

The final optimization layer identified 2,229 high-risk orders, allowing logistics teams to prioritize shipments requiring immediate intervention.

Overall, the project demonstrates how raw logistics data can be transformed into actionable intelligence for delivery monitoring, risk identification, resource allocation, and operational optimization.

👨‍💻 Author

Pottrivendhan

B.Tech Artificial Intelligence & Data Science

🔗 GitHub Repository

Logistics Data Analysis – Week 1 to Week 4

https://github.com/Pottrivendhan/logistics-data-analysis-week1

⭐ Project Status
Week 1  ✅ Completed
Week 2  ✅ Completed
Week 3  ✅ Completed
Week 4  ✅ Completed


Overall Project Status: ✅ COMPLETED
