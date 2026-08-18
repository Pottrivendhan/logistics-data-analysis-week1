import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA
# ==========================================

data_path = "data/"

orders = pd.read_csv(
    data_path + "olist_orders_dataset.csv"
)

items = pd.read_csv(
    data_path + "olist_order_items_dataset.csv"
)

print("Dataset loaded successfully!")
print("Total orders:", len(orders))


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

for col in date_columns:
    orders[col] = pd.to_datetime(
        orders[col],
        errors="coerce"
    )


# ==========================================
# 3. SELECT DELIVERED ORDERS
# ==========================================

delivered = orders[
    orders["order_delivered_customer_date"].notna()
].copy()

print("Delivered orders:", len(delivered))


# ==========================================
# 4. DELIVERY TIME
# ==========================================

delivered["delivery_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400


# ==========================================
# 5. LATE DELIVERY
# ==========================================

delivered["late_delivery"] = (
    delivered["order_delivered_customer_date"]
    > delivered["order_estimated_delivery_date"]
).astype(int)


# ==========================================
# 6. LOGISTICS KPIs
# ==========================================

average_delivery_days = delivered["delivery_days"].mean()

late_delivery_rate = (
    delivered["late_delivery"].mean() * 100
)

on_time_delivery_rate = 100 - late_delivery_rate


# ==========================================
# 7. FREIGHT COST
# ==========================================

freight_by_order = (
    items.groupby("order_id")["freight_value"]
    .sum()
)

average_freight_cost = freight_by_order.mean()


# ==========================================
# 8. DISPLAY KPIs
# ==========================================

print("\n================================")
print("       LOGISTICS KPIs")
print("================================")

print(
    "Average Delivery Time:",
    round(average_delivery_days, 2),
    "days"
)

print(
    "Late Delivery Rate:",
    round(late_delivery_rate, 2),
    "%"
)

print(
    "On-Time Delivery Rate:",
    round(on_time_delivery_rate, 2),
    "%"
)

print(
    "Average Freight Cost:",
    round(average_freight_cost, 2)
)

print("================================")


# ==========================================
# 9. SAVE CLEANED DATA
# ==========================================

delivered.to_csv(
    "cleaned_logistics_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


# ==========================================
# 10. DELIVERY TIME DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(
    delivered["delivery_days"],
    bins=30
)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()


# ==========================================
# 11. ON-TIME VS LATE
# ==========================================

late_counts = delivered["late_delivery"].value_counts()

plt.figure(figsize=(6, 5))

plt.bar(
    ["On Time", "Late"],
    [
        late_counts.get(0, 0),
        late_counts.get(1, 0)
    ]
)

plt.title("On-Time vs Late Deliveries")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()


# ==========================================
# 12. MONTHLY DELIVERY ANALYSIS
# ==========================================

delivered["month"] = (
    delivered["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

monthly_analysis = (
    delivered.groupby("month")
    .agg(
        total_orders=("order_id", "count"),
        late_orders=("late_delivery", "sum"),
        average_delivery_days=("delivery_days", "mean")
    )
)

monthly_analysis["late_rate"] = (
    monthly_analysis["late_orders"]
    / monthly_analysis["total_orders"]
    * 100
)

print("\n================================")
print("       MONTHLY ANALYSIS")
print("================================")

print(
    monthly_analysis.round(2)
)


# ==========================================
# 13. MONTHLY LATE DELIVERY GRAPH
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_analysis.index,
    monthly_analysis["late_rate"],
    marker="o"
)

plt.title("Monthly Late Delivery Rate")
plt.xlabel("Month")
plt.ylabel("Late Delivery Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==========================================
# 14. MONTHLY AVERAGE DELIVERY GRAPH
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_analysis.index,
    monthly_analysis["average_delivery_days"],
    marker="o"
)

plt.title("Monthly Average Delivery Time")
plt.xlabel("Month")
plt.ylabel("Average Delivery Time (Days)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ==========================================
# 15. SELLER PERFORMANCE ANALYSIS
# ==========================================

# Load seller information
sellers = pd.read_csv(
    data_path + "olist_sellers_dataset.csv"
)

# Connect orders with order items
seller_orders = items[
    ["order_id", "seller_id"]
].drop_duplicates()

# Merge seller information with delivered orders
seller_analysis = delivered[
    [
        "order_id",
        "delivery_days",
        "late_delivery"
    ]
].merge(
    seller_orders,
    on="order_id",
    how="inner"
)

# ==========================================
# 16. CALCULATE SELLER KPIs
# ==========================================

seller_performance = (
    seller_analysis
    .groupby("seller_id")
    .agg(
        total_orders=("order_id", "count"),
        average_delivery_days=("delivery_days", "mean"),
        late_orders=("late_delivery", "sum")
    )
)

seller_performance["late_rate"] = (
    seller_performance["late_orders"]
    / seller_performance["total_orders"]
    * 100
)

# Sort by number of orders
seller_performance = seller_performance.sort_values(
    "total_orders",
    ascending=False
)

print("\n========================================")
print("       TOP SELLERS BY ORDER VOLUME")
print("========================================")

print(
    seller_performance.head(10).round(2)
)


# ==========================================
# 17. HIGH-RISK SELLERS
# ==========================================

# Consider sellers with at least 20 delivered orders
# to avoid very small samples.

high_risk_sellers = seller_performance[
    seller_performance["total_orders"] >= 20
].sort_values(
    "late_rate",
    ascending=False
)

print("\n========================================")
print("       HIGH-RISK SELLERS")
print("========================================")

print(
    high_risk_sellers.head(10).round(2)
)


# ==========================================
# 18. SELLER DELIVERY TIME GRAPH
# ==========================================

top_sellers = seller_performance.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_sellers.index.astype(str),
    top_sellers["average_delivery_days"]
)

plt.title("Average Delivery Time - Top 10 Sellers")
plt.xlabel("Seller ID")
plt.ylabel("Average Delivery Time (Days)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==========================================
# 19. SELLER LATE RATE GRAPH
# ==========================================

top_risk = high_risk_sellers.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_risk.index.astype(str),
    top_risk["late_rate"]
)

plt.title("Highest Late Delivery Rate - Sellers")
plt.xlabel("Seller ID")
plt.ylabel("Late Delivery Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# ==========================================
# 20. REGIONAL DELIVERY ANALYSIS
# ==========================================

# Load customer dataset
customers = pd.read_csv(
    data_path + "olist_customers_dataset.csv"
)

# Select required customer information
customer_location = customers[
    [
        "customer_id",
        "customer_state",
        "customer_city"
    ]
].drop_duplicates()

# Connect delivered orders with customer location
regional_analysis = delivered[
    [
        "order_id",
        "customer_id",
        "delivery_days",
        "late_delivery"
    ]
].merge(
    customer_location,
    on="customer_id",
    how="left"
)


# ==========================================
# 21. STATE-LEVEL PERFORMANCE
# ==========================================

state_performance = (
    regional_analysis
    .groupby("customer_state")
    .agg(
        total_orders=("order_id", "count"),
        average_delivery_days=("delivery_days", "mean"),
        late_orders=("late_delivery", "sum")
    )
)

state_performance["late_rate"] = (
    state_performance["late_orders"]
    / state_performance["total_orders"]
    * 100
)

# Sort by order volume
state_performance = state_performance.sort_values(
    "total_orders",
    ascending=False
)


# ==========================================
# 22. DISPLAY STATE PERFORMANCE
# ==========================================

print("\n========================================")
print("       STATE DELIVERY PERFORMANCE")
print("========================================")

print(
    state_performance.round(2)
)


# ==========================================
# 23. STATES WITH HIGHEST LATE RATE
# ==========================================

# Minimum 100 orders to avoid small samples
high_risk_states = state_performance[
    state_performance["total_orders"] >= 100
].sort_values(
    "late_rate",
    ascending=False
)

print("\n========================================")
print("       HIGHEST-RISK STATES")
print("========================================")

print(
    high_risk_states.head(10).round(2)
)


# ==========================================
# 24. STATES WITH LONGEST DELIVERY TIME
# ==========================================

slowest_states = state_performance[
    state_performance["total_orders"] >= 100
].sort_values(
    "average_delivery_days",
    ascending=False
)

print("\n========================================")
print("       SLOWEST STATES")
print("========================================")

print(
    slowest_states.head(10).round(2)
)


# ==========================================
# 25. TOP STATES BY ORDER VOLUME
# ==========================================

top_states = state_performance.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_states.index,
    top_states["total_orders"]
)

plt.title("Top 10 States by Order Volume")
plt.xlabel("Customer State")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()


# ==========================================
# 26. HIGHEST LATE DELIVERY STATES
# ==========================================

risk_states = high_risk_states.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    risk_states.index,
    risk_states["late_rate"]
)

plt.title("Top 10 States by Late Delivery Rate")
plt.xlabel("Customer State")
plt.ylabel("Late Delivery Rate (%)")

plt.tight_layout()
plt.show()
# ==========================================
# 27. PRODUCT & FREIGHT ANALYSIS
# ==========================================

# Load product dataset
products = pd.read_csv(
    data_path + "olist_products_dataset.csv"
)

# Select useful product columns
product_info = products[
    [
        "product_id",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]
]

# Add product information to order items
product_orders = items[
    [
        "order_id",
        "product_id",
        "price",
        "freight_value"
    ]
].merge(
    product_info,
    on="product_id",
    how="left"
)

# Add delivery information
product_delivery = product_orders.merge(
    delivered[
        [
            "order_id",
            "delivery_days",
            "late_delivery"
        ]
    ],
    on="order_id",
    how="inner"
)


# ==========================================
# 28. PRODUCT-LEVEL SUMMARY
# ==========================================

product_summary = (
    product_delivery
    .groupby("product_id")
    .agg(
        order_count=("order_id", "count"),
        average_price=("price", "mean"),
        average_freight=("freight_value", "mean"),
        average_delivery_days=("delivery_days", "mean"),
        late_rate=("late_delivery", "mean"),
        average_weight=("product_weight_g", "mean")
    )
)

product_summary["late_rate"] = (
    product_summary["late_rate"] * 100
)


print("\n========================================")
print("       PRODUCT & FREIGHT ANALYSIS")
print("========================================")

print(
    product_summary.head(10).round(2)
)


# ==========================================
# 29. FREIGHT VS DELIVERY TIME
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    product_delivery["freight_value"],
    product_delivery["delivery_days"],
    alpha=0.3
)

plt.title("Freight Cost vs Delivery Time")
plt.xlabel("Freight Cost")
plt.ylabel("Delivery Time (Days)")

plt.tight_layout()
plt.show()


# ==========================================
# 30. PRODUCT WEIGHT VS FREIGHT
# ==========================================

weight_data = product_delivery.dropna(
    subset=[
        "product_weight_g",
        "freight_value"
    ]
)

plt.figure(figsize=(10, 6))

plt.scatter(
    weight_data["product_weight_g"],
    weight_data["freight_value"],
    alpha=0.3
)

plt.title("Product Weight vs Freight Cost")
plt.xlabel("Product Weight (grams)")
plt.ylabel("Freight Cost")

plt.tight_layout()
plt.show()


# ==========================================
# 31. PRODUCT WEIGHT VS DELIVERY TIME
# ==========================================

delivery_weight = product_delivery.dropna(
    subset=[
        "product_weight_g",
        "delivery_days"
    ]
)

plt.figure(figsize=(10, 6))

plt.scatter(
    delivery_weight["product_weight_g"],
    delivery_weight["delivery_days"],
    alpha=0.3
)

plt.title("Product Weight vs Delivery Time")
plt.xlabel("Product Weight (grams)")
plt.ylabel("Delivery Time (Days)")

plt.tight_layout()
plt.show()


# ==========================================
# 32. FREIGHT STATISTICS
# ==========================================

print("\n========================================")
print("       FREIGHT STATISTICS")
print("========================================")

print(
    "Average Freight:",
    round(product_delivery["freight_value"].mean(), 2)
)

print(
    "Maximum Freight:",
    round(product_delivery["freight_value"].max(), 2)
)

print(
    "Minimum Freight:",
    round(product_delivery["freight_value"].min(), 2)
)

print(
    "Average Product Weight:",
    round(
        product_delivery["product_weight_g"].mean(),
        2
    ),
    "grams"
)
# ==========================================
# 33. PREDICTIVE MODELING
#     LATE DELIVERY PREDICTION
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 34. PREPARE CUSTOMER INFORMATION
# ==========================================

customer_features = customers[
    [
        "customer_id",
        "customer_state"
    ]
].drop_duplicates()


# ==========================================
# 35. PREPARE ORDER-ITEM FEATURES
# ==========================================

item_features = (
    items.groupby("order_id")
    .agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        average_freight=("freight_value", "mean")
    )
    .reset_index()
)


# ==========================================
# 36. GET SELLER INFORMATION
# ==========================================

seller_features = (
    items[
        [
            "order_id",
            "seller_id"
        ]
    ]
    .drop_duplicates("order_id")
)


# ==========================================
# 37. CREATE MODEL DATASET
# ==========================================

model_data = delivered[
    [
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "late_delivery"
    ]
].copy()


# Add customer state
model_data = model_data.merge(
    customer_features,
    on="customer_id",
    how="left"
)


# Add seller
model_data = model_data.merge(
    seller_features,
    on="order_id",
    how="left"
)


# Add item and freight features
model_data = model_data.merge(
    item_features,
    on="order_id",
    how="left"
)


# ==========================================
# 38. CREATE TIME FEATURES
# ==========================================

model_data["purchase_month"] = (
    model_data["order_purchase_timestamp"]
    .dt.month
)

model_data["purchase_day"] = (
    model_data["order_purchase_timestamp"]
    .dt.dayofweek
)


# ==========================================
# 39. SELECT FEATURES
# ==========================================

features = [
    "customer_state",
    "seller_id",
    "item_count",
    "total_price",
    "total_freight",
    "average_freight",
    "purchase_month",
    "purchase_day"
]

X = model_data[features]

y = model_data["late_delivery"]


# ==========================================
# 40. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 41. PREPROCESSING
# ==========================================

categorical_features = [
    "customer_state",
    "seller_id"
]

numeric_features = [
    "item_count",
    "total_price",
    "total_freight",
    "average_freight",
    "purchase_month",
    "purchase_day"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
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
            ]),
            categorical_features
        ),
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]),
            numeric_features
        )
    ]
)


# ==========================================
# 42. RANDOM FOREST MODEL
# ==========================================

model = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])


# ==========================================
# 43. TRAIN MODEL
# ==========================================

print("\n========================================")
print("       TRAINING MODEL")
print("========================================")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ==========================================
# 44. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 45. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("       MODEL RESULTS")
print("========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 46. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 47. CONFUSION MATRIX GRAPH
# ==========================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Late Delivery Prediction - Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["On Time", "Late"]
)

plt.yticks(
    [0, 1],
    ["On Time", "Late"]
)

plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()
# ==========================================
# 48. SELLER CLUSTERING
# ==========================================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==========================================
# 49. CREATE SELLER FEATURES
# ==========================================

seller_clustering = (
    seller_analysis
    .groupby("seller_id")
    .agg(
        order_volume=("order_id", "count"),
        average_delivery_days=("delivery_days", "mean"),
        late_rate=("late_delivery", "mean")
    )
)


# Convert late rate to percentage
seller_clustering["late_rate"] = (
    seller_clustering["late_rate"] * 100
)


print("\n========================================")
print("       SELLER CLUSTERING DATA")
print("========================================")

print(
    seller_clustering.head(10).round(2)
)


# ==========================================
# 50. SELECT CLUSTERING FEATURES
# ==========================================

cluster_features = [
    "order_volume",
    "average_delivery_days",
    "late_rate"
]

X_cluster = seller_clustering[
    cluster_features
].copy()


# ==========================================
# 51. STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_cluster
)


# ==========================================
# 52. APPLY K-MEANS
# ==========================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

seller_clustering["cluster"] = (
    kmeans.fit_predict(X_scaled)
)


# ==========================================
# 53. CLUSTER SUMMARY
# ==========================================

cluster_summary = (
    seller_clustering
    .groupby("cluster")
    .agg(
        sellers=("order_volume", "count"),
        average_orders=("order_volume", "mean"),
        average_delivery_days=(
            "average_delivery_days",
            "mean"
        ),
        average_late_rate=(
            "late_rate",
            "mean"
        )
    )
)


print("\n========================================")
print("       CLUSTER SUMMARY")
print("========================================")

print("\nComplete Cluster Summary:")

print(
    cluster_summary.to_string()
)


# ==========================================
# 54. CLUSTER VISUALIZATION
# ==========================================

plt.figure(figsize=(10, 6))

for cluster in sorted(
    seller_clustering["cluster"].unique()
):

    cluster_data = seller_clustering[
        seller_clustering["cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["average_delivery_days"],
        cluster_data["late_rate"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )


plt.title(
    "Seller Logistics Performance Clusters"
)

plt.xlabel(
    "Average Delivery Time (Days)"
)

plt.ylabel(
    "Late Delivery Rate (%)"
)

plt.legend()

plt.tight_layout()
plt.show()