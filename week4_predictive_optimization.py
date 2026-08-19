import pandas as pd
import numpy as np

print("========================================")
print(" WEEK 4 - PREDICTIVE MODELING")
print("========================================")

# ==========================================
# 1. LOAD DATA
# ==========================================

orders = pd.read_csv(
    "data/olist_orders_dataset.csv"
)

order_items = pd.read_csv(
    "data/olist_order_items_dataset.csv"
)

customers = pd.read_csv(
    "data/olist_customers_dataset.csv"
)

products = pd.read_csv(
    "data/olist_products_dataset.csv"
)

print("\nDatasets loaded successfully!")

print("\nOrders:", orders.shape)
print("Order Items:", order_items.shape)
print("Customers:", customers.shape)
print("Products:", products.shape)


# ==========================================
# 2. CONVERT DATE COLUMNS
# ==========================================

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column],
        errors="coerce"
    )


# ==========================================
# 3. SELECT DELIVERED ORDERS
# ==========================================

delivered = orders[
    orders["order_status"] == "delivered"
].copy()

print("\nDelivered orders:", len(delivered))


# ==========================================
# 4. CREATE TARGET VARIABLE
# ==========================================

delivered["delivery_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400


# ==========================================
# 5. CREATE LATE DELIVERY FEATURE
# ==========================================

delivered["late_delivery"] = (
    delivered["order_delivered_customer_date"]
    > delivered["order_estimated_delivery_date"]
).astype(int)


# ==========================================
# 6. MERGE ORDER ITEMS
# ==========================================

items_summary = order_items.groupby(
    "order_id"
).agg(
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    average_price=("price", "mean"),
    average_freight=("freight_value", "mean"),
    item_count=("order_item_id", "count")
).reset_index()

delivered = delivered.merge(
    items_summary,
    on="order_id",
    how="left"
)


# ==========================================
# 7. MERGE CUSTOMER INFORMATION
# ==========================================

customer_info = customers[
    [
        "customer_id",
        "customer_state"
    ]
].drop_duplicates()

delivered = delivered.merge(
    customer_info,
    on="customer_id",
    how="left"
)


# ==========================================
# 8. FEATURE ENGINEERING
# ==========================================

delivered["order_month"] = (
    delivered["order_purchase_timestamp"]
    .dt.month
)

delivered["order_dayofweek"] = (
    delivered["order_purchase_timestamp"]
    .dt.dayofweek
)

delivered["order_year"] = (
    delivered["order_purchase_timestamp"]
    .dt.year
)


# ==========================================
# 9. REMOVE INVALID TARGET VALUES
# ==========================================

delivered = delivered[
    delivered["delivery_days"].notna()
]

delivered = delivered[
    delivered["delivery_days"] >= 0
]


# ==========================================
# 10. DISPLAY DATA
# ==========================================

print("\n========================================")
print(" WEEK 4 MODELING DATA")
print("========================================")

print(
    delivered[
        [
            "delivery_days",
            "total_price",
            "total_freight",
            "average_price",
            "average_freight",
            "item_count",
            "customer_state",
            "order_month",
            "order_dayofweek"
        ]
    ].head()
)

print("\nModeling dataset shape:")
print(delivered.shape)

print("\nMissing values:")
print(
    delivered.isnull().sum()
)

# ==========================================
# 11. SAVE MODELING DATA
# ==========================================

delivered.to_csv(
    "week4_modeling_data.csv",
    index=False
)

print("\nModeling dataset saved successfully!")

print("\nSTEP 1 COMPLETED!")
# ==========================================
# WEEK 4 - STEP 2
# PREPROCESSING AND TRAIN/TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


print("\n========================================")
print(" STEP 2 - PREPROCESSING")
print("========================================")


# ==========================================
# 1. SELECT FEATURES
# ==========================================

features = [
    "total_price",
    "total_freight",
    "average_price",
    "average_freight",
    "item_count",
    "customer_state",
    "order_month",
    "order_dayofweek",
    "order_year"
]

target = "delivery_days"


X = delivered[features].copy()

y = delivered[target].copy()


print("\nFeatures selected:")
print(features)

print("\nTarget variable:")
print(target)


# ==========================================
# 2. DEFINE NUMERICAL FEATURES
# ==========================================

numeric_features = [
    "total_price",
    "total_freight",
    "average_price",
    "average_freight",
    "item_count",
    "order_month",
    "order_dayofweek",
    "order_year"
]


# ==========================================
# 3. DEFINE CATEGORICAL FEATURES
# ==========================================

categorical_features = [
    "customer_state"
]


# ==========================================
# 4. NUMERICAL PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ==========================================
# 5. CATEGORICAL PREPROCESSING
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ==========================================
# 6. COMBINE PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n========================================")
print(" TRAIN / TEST SPLIT")
print("========================================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Training percentage:",
    round(len(X_train) / len(X) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(X_test) / len(X) * 100, 2),
    "%"
)


# ==========================================
# 8. FIT PREPROCESSOR
# ==========================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print("\n========================================")
print(" PREPROCESSING COMPLETED")
print("========================================")

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


print("\nSTEP 2 COMPLETED!")
# ==========================================
# WEEK 4 - STEP 3
# PREDICTIVE MODEL TRAINING
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("\n========================================")
print(" STEP 3 - MODEL TRAINING")
print("========================================")


# ==========================================
# 1. LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_processed,
    y_train
)

linear_predictions = linear_model.predict(
    X_test_processed
)


# ==========================================
# 2. RANDOM FOREST
# ==========================================

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train_processed,
    y_train
)

random_forest_predictions = (
    random_forest_model.predict(
        X_test_processed
    )
)


print("\nModels trained successfully!")


# ==========================================
# 3. LINEAR REGRESSION METRICS
# ==========================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# ==========================================
# 4. RANDOM FOREST METRICS
# ==========================================

rf_mae = mean_absolute_error(
    y_test,
    random_forest_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        random_forest_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    random_forest_predictions
)


# ==========================================
# 5. DISPLAY RESULTS
# ==========================================

print("\n========================================")
print(" MODEL PERFORMANCE")
print("========================================")

print("\nLinear Regression")
print("----------------------------------------")

print(
    "MAE:",
    round(linear_mae, 4)
)

print(
    "RMSE:",
    round(linear_rmse, 4)
)

print(
    "R2 Score:",
    round(linear_r2, 4)
)


print("\nRandom Forest Regressor")
print("----------------------------------------")

print(
    "MAE:",
    round(rf_mae, 4)
)

print(
    "RMSE:",
    round(rf_rmse, 4)
)

print(
    "R2 Score:",
    round(rf_r2, 4)
)


# ==========================================
# 6. MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2
    ]
})


print("\n========================================")
print(" MODEL COMPARISON")
print("========================================")

print(
    comparison.round(4)
)


# ==========================================
# 7. SELECT BEST MODEL
# ==========================================

if rf_rmse < linear_rmse:

    best_model = random_forest_model
    best_predictions = random_forest_predictions
    best_model_name = "Random Forest"

else:

    best_model = linear_model
    best_predictions = linear_predictions
    best_model_name = "Linear Regression"


print("\n========================================")
print(" BEST MODEL")
print("========================================")

print(
    "Selected model:",
    best_model_name
)


# ==========================================
# 8. SAVE MODEL RESULTS
# ==========================================

comparison.to_csv(
    "week4_model_comparison.csv",
    index=False
)

print(
    "\nModel comparison saved successfully!"
)

print("\nSTEP 3 COMPLETED!")
# ==========================================
# WEEK 4 - STEP 4
# CROSS-VALIDATION & HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import GridSearchCV

print("\n========================================")
print(" STEP 4 - MODEL TUNING")
print("========================================")


# ==========================================
# 1. DEFINE RANDOM FOREST
# ==========================================

rf_base = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 2. DEFINE PARAMETERS
# ==========================================

param_grid = {
    "n_estimators": [100],
    "max_depth": [10, 15],
    "min_samples_split": [2, 5]
}


# ==========================================
# 3. GRID SEARCH
# ==========================================

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=2,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)


print("\nStarting Grid Search...")
print("Please wait...")


grid_search.fit(
    X_train_processed,
    y_train
)


# ==========================================
# 4. BEST PARAMETERS
# ==========================================

print("\n========================================")
print(" BEST PARAMETERS")
print("========================================")

print(
    grid_search.best_params_
)


print("\nBest Cross-Validation RMSE:")

print(
    round(
        -grid_search.best_score_,
        4
    )
)


# ==========================================
# 5. BEST MODEL
# ==========================================

tuned_model = grid_search.best_estimator_


# ==========================================
# 6. PREDICTIONS
# ==========================================

tuned_predictions = tuned_model.predict(
    X_test_processed
)


# ==========================================
# 7. EVALUATION
# ==========================================

tuned_mae = mean_absolute_error(
    y_test,
    tuned_predictions
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tuned_predictions
    )
)

tuned_r2 = r2_score(
    y_test,
    tuned_predictions
)


print("\n========================================")
print(" TUNED RANDOM FOREST RESULTS")
print("========================================")

print(
    "MAE:",
    round(tuned_mae, 4)
)

print(
    "RMSE:",
    round(tuned_rmse, 4)
)

print(
    "R2 Score:",
    round(tuned_r2, 4)
)


# ==========================================
# 8. COMPARE MODELS
# ==========================================

tuning_comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest",
        "Tuned Random Forest"
    ],

    "MAE": [
        linear_mae,
        rf_mae,
        tuned_mae
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse,
        tuned_rmse
    ],

    "R2": [
        linear_r2,
        rf_r2,
        tuned_r2
    ]
})


print("\n========================================")
print(" FINAL MODEL COMPARISON")
print("========================================")

print(
    tuning_comparison.round(4)
)


# ==========================================
# 9. SELECT FINAL MODEL
# ==========================================

if tuned_rmse < rf_rmse:

    final_model = tuned_model
    final_predictions = tuned_predictions
    final_model_name = "Tuned Random Forest"

else:

    final_model = random_forest_model
    final_predictions = random_forest_predictions
    final_model_name = "Random Forest"


print("\n========================================")
print(" FINAL MODEL")
print("========================================")

print(
    "Selected:",
    final_model_name
)


# ==========================================
# 10. SAVE RESULTS
# ==========================================

tuning_comparison.to_csv(
    "week4_final_model_comparison.csv",
    index=False
)

print(
    "\nFinal model comparison saved successfully!"
)

print("\nSTEP 4 COMPLETED!")

# ==========================================
# WEEK 4 - STEP 5
# DELIVERY TIME PREDICTION & RISK ANALYSIS
# ==========================================

print("\n========================================")
print(" STEP 5 - PREDICTION & RISK ANALYSIS")
print("========================================")


# ==========================================
# 1. CREATE PREDICTION DATA
# ==========================================

prediction_results = X_test.copy()

prediction_results["actual_delivery_days"] = (
    y_test.values
)

prediction_results["predicted_delivery_days"] = (
    final_predictions
)


# ==========================================
# 2. CALCULATE PREDICTION ERROR
# ==========================================

prediction_results["prediction_error"] = abs(
    prediction_results["actual_delivery_days"]
    - prediction_results["predicted_delivery_days"]
)


# ==========================================
# 3. CREATE RISK LEVEL
# ==========================================

def classify_risk(days):

    if days <= 10:
        return "Low Risk"

    elif days <= 20:
        return "Medium Risk"

    else:
        return "High Risk"


prediction_results["risk_level"] = (
    prediction_results[
        "predicted_delivery_days"
    ].apply(classify_risk)
)


# ==========================================
# 4. DISPLAY PREDICTIONS
# ==========================================

print("\n========================================")
print(" SAMPLE PREDICTIONS")
print("========================================")

print(
    prediction_results[
        [
            "actual_delivery_days",
            "predicted_delivery_days",
            "prediction_error",
            "risk_level"
        ]
    ].head(10).round(2)
)


# ==========================================
# 5. RISK DISTRIBUTION
# ==========================================

print("\n========================================")
print(" RISK DISTRIBUTION")
print("========================================")

risk_counts = (
    prediction_results["risk_level"]
    .value_counts()
)

print(risk_counts)


# ==========================================
# 6. RISK PERCENTAGE
# ==========================================

risk_percentage = (
    prediction_results["risk_level"]
    .value_counts(normalize=True)
    * 100
)

print("\nRisk Percentage:")

print(
    risk_percentage.round(2)
)


# ==========================================
# 7. HIGH-RISK ORDERS
# ==========================================

high_risk_orders = prediction_results[
    prediction_results["risk_level"] == "High Risk"
].copy()


print("\n========================================")
print(" HIGH-RISK ORDERS")
print("========================================")

print(
    "Number of high-risk orders:",
    len(high_risk_orders)
)


# ==========================================
# 8. SAVE PREDICTIONS
# ==========================================

prediction_results.to_csv(
    "week4_delivery_predictions.csv",
    index=False
)


high_risk_orders.to_csv(
    "week4_high_risk_orders.csv",
    index=False
)


print("\nPrediction results saved successfully!")

print(
    "week4_delivery_predictions.csv"
)

print(
    "week4_high_risk_orders.csv"
)


print("\nSTEP 5 COMPLETED!")

# ==========================================
# WEEK 4 - STEP 6
# LOGISTICS OPTIMIZATION ANALYSIS
# ==========================================

print("\n========================================")
print(" STEP 6 - LOGISTICS OPTIMIZATION")
print("========================================")


# ==========================================
# 1. RISK SUMMARY
# ==========================================

optimization_summary = (
    prediction_results
    .groupby("risk_level")
    .agg(
        order_count=("predicted_delivery_days", "count"),
        average_predicted_days=(
            "predicted_delivery_days",
            "mean"
        ),
        average_actual_days=(
            "actual_delivery_days",
            "mean"
        ),
        average_prediction_error=(
            "prediction_error",
            "mean"
        )
    )
    .reset_index()
)


print("\n========================================")
print(" OPTIMIZATION SUMMARY")
print("========================================")

print(
    optimization_summary.round(2)
)


# ==========================================
# 2. HIGH-RISK ANALYSIS
# ==========================================

high_risk_summary = (
    prediction_results[
        prediction_results["risk_level"] == "High Risk"
    ]
    .agg(
        order_count=(
            "predicted_delivery_days",
            "count"
        ),
        average_predicted_days=(
            "predicted_delivery_days",
            "mean"
        ),
        average_actual_days=(
            "actual_delivery_days",
            "mean"
        ),
        average_error=(
            "prediction_error",
            "mean"
        )
    )
)


print("\n========================================")
print(" HIGH-RISK OPTIMIZATION")
print("========================================")

print(
    high_risk_summary.round(2)
)


# ==========================================
# 3. PRIORITY SCORE
# ==========================================

prediction_results["priority_score"] = (
    prediction_results[
        "predicted_delivery_days"
    ] / prediction_results[
        "predicted_delivery_days"
    ].max()
) * 100


# ==========================================
# 4. TOP PRIORITY ORDERS
# ==========================================

priority_orders = (
    prediction_results
    .sort_values(
        "priority_score",
        ascending=False
    )
    .head(20)
)


print("\n========================================")
print(" TOP 20 PRIORITY ORDERS")
print("========================================")

print(
    priority_orders[
        [
            "predicted_delivery_days",
            "actual_delivery_days",
            "risk_level",
            "priority_score"
        ]
    ].round(2)
)


# ==========================================
# 5. OPTIMIZATION ACTION
# ==========================================

def optimization_action(risk):

    if risk == "High Risk":
        return "Immediate intervention"

    elif risk == "Medium Risk":
        return "Monitor shipment"

    else:
        return "Normal processing"


prediction_results["recommended_action"] = (
    prediction_results[
        "risk_level"
    ].apply(optimization_action)
)


# ==========================================
# 6. ACTION DISTRIBUTION
# ==========================================

print("\n========================================")
print(" RECOMMENDED ACTIONS")
print("========================================")

print(
    prediction_results[
        "recommended_action"
    ].value_counts()
)


# ==========================================
# 7. SAVE OPTIMIZATION RESULTS
# ==========================================

optimization_summary.to_csv(
    "week4_optimization_summary.csv",
    index=False
)

priority_orders.to_csv(
    "week4_priority_orders.csv",
    index=False
)

prediction_results.to_csv(
    "week4_final_predictions.csv",
    index=False
)


print("\nOptimization files saved successfully!")

print("week4_optimization_summary.csv")
print("week4_priority_orders.csv")
print("week4_final_predictions.csv")


print("\nSTEP 6 COMPLETED!")