import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# WEEK 3
# ADVANCED DATA ANALYSIS & VISUALIZATION
# ==========================================

print("========================================")
print(" WEEK 3 DATA LOADING")
print("========================================")

# Load datasets
orders = pd.read_csv(
    "data/olist_orders_dataset.csv"
)

order_items = pd.read_csv(
    "data/olist_order_items_dataset.csv"
)

customers = pd.read_csv(
    "data/olist_customers_dataset.csv"
)

sellers = pd.read_csv(
    "data/olist_sellers_dataset.csv"
)

products = pd.read_csv(
    "data/olist_products_dataset.csv"
)


# ==========================================
# DATE CONVERSION
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
# DELIVERED ORDERS
# ==========================================

delivered = orders[
    orders["order_status"] == "delivered"
].copy()


# ==========================================
# DELIVERY TIME
# ==========================================

delivered["delivery_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400


# ==========================================
# LATE DELIVERY
# ==========================================

delivered["late_delivery"] = (
    delivered["order_delivered_customer_date"]
    > delivered["order_estimated_delivery_date"]
).astype(int)


print("\nTotal orders:", len(orders))

print(
    "Delivered orders:",
    len(delivered)
)

print(
    "Average delivery time:",
    round(
        delivered["delivery_days"].mean(),
        2
    ),
    "days"
)

print(
    "Late delivery rate:",
    round(
        delivered["late_delivery"].mean() * 100,
        2
    ),
    "%"
)


# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

print("\n========================================")
print(" DESCRIPTIVE STATISTICS")
print("========================================")

print(
    delivered["delivery_days"].describe()
)


print("\nOrder Item Statistics:")

print(
    order_items[
        ["price", "freight_value"]
    ].describe()
)


print("\nProduct Statistics:")

print(
    products[
        [
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm"
        ]
    ].describe()
)


# ==========================================
# CORRELATION ANALYSIS
# ==========================================

print("\n========================================")
print(" CORRELATION ANALYSIS")
print("========================================")

correlation_data = order_items[
    ["price", "freight_value"]
].corr()

print(correlation_data)


# ==========================================
# MONTHLY ANALYSIS
# ==========================================

delivered["month"] = (
    delivered[
        "order_purchase_timestamp"
    ].dt.to_period("M")
    .astype(str)
)

monthly = delivered.groupby(
    "month"
).agg(
    total_orders=("order_id", "count"),
    average_delivery_days=(
        "delivery_days",
        "mean"
    ),
    late_rate=(
        "late_delivery",
        "mean"
    )
)

monthly["late_rate"] *= 100

print("\n========================================")
print(" MONTHLY ANALYSIS")
print("========================================")

print(monthly)


# ==========================================
# CREATE CHART FOLDER
# ==========================================

import os

os.makedirs(
    "week3_charts",
    exist_ok=True
)


# ==========================================
# CHART 1
# DELIVERY TIME DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(
    delivered["delivery_days"].dropna(),
    bins=40
)

plt.title(
    "Distribution of Delivery Time"
)

plt.xlabel(
    "Delivery Time (Days)"
)

plt.ylabel(
    "Number of Orders"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/delivery_time_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# CHART 2
# MONTHLY ORDER VOLUME
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly.index,
    monthly["total_orders"],
    marker="o"
)

plt.title(
    "Monthly Order Volume"
)

plt.xlabel("Month")

plt.ylabel("Number of Orders")

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "week3_charts/monthly_order_volume.png",
    dpi=300
)

plt.close()


# ==========================================
# CHART 3
# MONTHLY LATE DELIVERY RATE
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly.index,
    monthly["late_rate"],
    marker="o"
)

plt.title(
    "Monthly Late Delivery Rate"
)

plt.xlabel("Month")

plt.ylabel(
    "Late Delivery Rate (%)"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "week3_charts/monthly_late_rate.png",
    dpi=300
)

plt.close()


# ==========================================
# CHART 4
# FREIGHT DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(
    order_items["freight_value"],
    bins=40
)

plt.title(
    "Distribution of Freight Value"
)

plt.xlabel(
    "Freight Value"
)

plt.ylabel(
    "Number of Items"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/freight_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# CHART 5
# PRICE VS FREIGHT
# ==========================================

sample = order_items.sample(
    min(10000, len(order_items)),
    random_state=42
)

plt.figure(figsize=(10, 6))

plt.scatter(
    sample["price"],
    sample["freight_value"],
    alpha=0.4
)

plt.title(
    "Product Price vs Freight Value"
)

plt.xlabel(
    "Product Price"
)

plt.ylabel(
    "Freight Value"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/price_vs_freight.png",
    dpi=300
)

plt.close()


# ==========================================
# CHART 6
# PRODUCT WEIGHT VS FREIGHT
# ==========================================

merged_items = order_items.merge(
    products[
        [
            "product_id",
            "product_weight_g"
        ]
    ],
    on="product_id",
    how="left"
)

sample_weight = merged_items.sample(
    min(10000, len(merged_items)),
    random_state=42
)

plt.figure(figsize=(10, 6))

plt.scatter(
    sample_weight["product_weight_g"],
    sample_weight["freight_value"],
    alpha=0.4
)

plt.title(
    "Product Weight vs Freight Value"
)

plt.xlabel(
    "Product Weight (grams)"
)

plt.ylabel(
    "Freight Value"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/weight_vs_freight.png",
    dpi=300
)

plt.close()


print("\n========================================")
print(" VISUALIZATION COMPLETED")
print("========================================")

print(
    "Charts saved in: week3_charts/"
)
# ==========================================
# WEEK 3 - STEP 2
# ADVANCED LOGISTICS ANALYSIS
# ==========================================

print("\n========================================")
print(" ADVANCED LOGISTICS ANALYSIS")
print("========================================")


# ==========================================
# 1. DELIVERY PERFORMANCE BY STATE
# ==========================================

customer_state = customers[
    [
        "customer_id",
        "customer_state"
    ]
].drop_duplicates()

state_data = delivered.merge(
    customer_state,
    on="customer_id",
    how="left"
)

state_analysis = state_data.groupby(
    "customer_state"
).agg(
    total_orders=("order_id", "count"),
    average_delivery_days=(
        "delivery_days",
        "mean"
    ),
    late_rate=(
        "late_delivery",
        "mean"
    )
)

state_analysis["late_rate"] *= 100

state_analysis = state_analysis.sort_values(
    "late_rate",
    ascending=False
)

print("\n========================================")
print(" STATE DELIVERY PERFORMANCE")
print("========================================")

print(
    state_analysis.round(2)
)


# ==========================================
# 2. TOP HIGH-RISK STATES
# ==========================================

high_risk_states = state_analysis[
    state_analysis["total_orders"] >= 100
].head(10)

print("\n========================================")
print(" TOP HIGH-RISK STATES")
print("========================================")

print(
    high_risk_states.round(2)
)


# ==========================================
# 3. SELLER PERFORMANCE
# ==========================================

seller_items = order_items[
    [
        "order_id",
        "seller_id",
        "price",
        "freight_value"
    ]
]

seller_analysis_data = delivered.merge(
    seller_items,
    on="order_id",
    how="inner"
)

seller_analysis = seller_analysis_data.groupby(
    "seller_id"
).agg(
    total_orders=("order_id", "count"),
    average_delivery_days=(
        "delivery_days",
        "mean"
    ),
    late_rate=(
        "late_delivery",
        "mean"
    ),
    average_price=("price", "mean"),
    average_freight=("freight_value", "mean")
)

seller_analysis["late_rate"] *= 100

# Require at least 20 orders
reliable_sellers = seller_analysis[
    seller_analysis["total_orders"] >= 20
].copy()

reliable_sellers = reliable_sellers.sort_values(
    "late_rate",
    ascending=False
)

print("\n========================================")
print(" HIGH-RISK SELLERS")
print("========================================")

print(
    reliable_sellers.head(10).round(2)
)


# ==========================================
# 4. FREIGHT COST ANALYSIS
# ==========================================

freight_summary = order_items[
    "freight_value"
].describe()

print("\n========================================")
print(" FREIGHT COST ANALYSIS")
print("========================================")

print(
    freight_summary.round(2)
)


# ==========================================
# 5. PRICE-FREIGHT CORRELATION
# ==========================================

price_freight_corr = order_items[
    [
        "price",
        "freight_value"
    ]
].corr().iloc[0, 1]

print(
    "\nPrice vs Freight correlation:",
    round(price_freight_corr, 3)
)


# ==========================================
# 6. DELIVERY TIME CORRELATION
# ==========================================

print("\n========================================")
print(" DELIVERY TIME STATISTICS")
print("========================================")

print(
    delivered["delivery_days"].describe().round(2)
)


# ==========================================
# 7. BEST AND WORST MONTHS
# ==========================================

best_month = monthly[
    "late_rate"
].idxmin()

worst_month = monthly[
    "late_rate"
].idxmax()

print("\n========================================")
print(" MONTHLY PERFORMANCE")
print("========================================")

print(
    "Best month:",
    best_month,
    "-",
    round(
        monthly.loc[
            best_month,
            "late_rate"
        ],
        2
    ),
    "%"
)

print(
    "Worst month:",
    worst_month,
    "-",
    round(
        monthly.loc[
            worst_month,
            "late_rate"
        ],
        2
    ),
    "%"
)


# ==========================================
# 8. STATE VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 7))

plot_states = state_analysis[
    state_analysis["total_orders"] >= 100
].head(10)

plt.bar(
    plot_states.index,
    plot_states["late_rate"]
)

plt.title(
    "Top 10 States by Late Delivery Rate"
)

plt.xlabel("State")

plt.ylabel(
    "Late Delivery Rate (%)"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "week3_charts/state_late_delivery_rate.png",
    dpi=300
)

plt.close()


# ==========================================
# 9. TOP SELLER VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 7))

top_sellers = reliable_sellers.head(10)

plt.bar(
    range(len(top_sellers)),
    top_sellers["late_rate"]
)

plt.title(
    "High-Risk Sellers by Late Delivery Rate"
)

plt.xlabel("Seller")

plt.ylabel(
    "Late Delivery Rate (%)"
)

plt.xticks(
    range(len(top_sellers)),
    [
        seller[:8]
        for seller in top_sellers.index
    ],
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "week3_charts/high_risk_sellers.png",
    dpi=300
)

plt.close()


# ==========================================
# 10. SAVE ANALYSIS TABLES
# ==========================================

state_analysis.to_csv(
    "week3_state_analysis.csv"
)

seller_analysis.to_csv(
    "week3_seller_analysis.csv"
)

monthly.to_csv(
    "week3_monthly_analysis.csv"
)


print("\n========================================")
print(" ADVANCED ANALYSIS COMPLETED")
print("========================================")

print("\nFiles generated:")

print("week3_state_analysis.csv")
print("week3_seller_analysis.csv")
print("week3_monthly_analysis.csv")

print("\nAdditional charts:")

print(
    "state_late_delivery_rate.png"
)

print(
    "high_risk_sellers.png"
)
# ==========================================
# WEEK 3 - STEP 3
# CORRELATION & ADVANCED VISUALIZATION
# ==========================================

print("\n========================================")
print(" CORRELATION & ADVANCED VISUALIZATION")
print("========================================")


# ==========================================
# 1. PREPARE CORRELATION DATA
# ==========================================

correlation_df = pd.DataFrame({
    "delivery_days": delivered["delivery_days"],
    "late_delivery": delivered["late_delivery"]
})

# Add order-item information
delivery_items = delivered[
    [
        "order_id",
        "delivery_days",
        "late_delivery"
    ]
].merge(
    order_items[
        [
            "order_id",
            "price",
            "freight_value"
        ]
    ],
    on="order_id",
    how="inner"
)


# ==========================================
# 2. CORRELATION MATRIX
# ==========================================

correlation_matrix = delivery_items[
    [
        "delivery_days",
        "late_delivery",
        "price",
        "freight_value"
    ]
].corr()


print("\n========================================")
print(" CORRELATION MATRIX")
print("========================================")

print(
    correlation_matrix.round(3)
)


# ==========================================
# 3. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title(
    "Logistics Feature Correlation Heatmap"
)

# Add correlation values
for i in range(
    len(correlation_matrix.columns)
):

    for j in range(
        len(correlation_matrix.columns)
    ):

        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "week3_charts/correlation_heatmap.png",
    dpi=300
)

plt.close()


# ==========================================
# 4. DELIVERY TIME VS FREIGHT
# ==========================================

delivery_freight_sample = delivery_items.sample(
    min(10000, len(delivery_items)),
    random_state=42
)

plt.figure(figsize=(10, 6))

plt.scatter(
    delivery_freight_sample["freight_value"],
    delivery_freight_sample["delivery_days"],
    alpha=0.4
)

plt.title(
    "Freight Value vs Delivery Time"
)

plt.xlabel(
    "Freight Value"
)

plt.ylabel(
    "Delivery Time (Days)"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/freight_vs_delivery_time.png",
    dpi=300
)

plt.close()


# ==========================================
# 5. ON-TIME VS LATE ORDERS
# ==========================================

delivery_status = (
    delivered["late_delivery"]
    .map({
        0: "On-Time",
        1: "Late"
    })
    .value_counts()
)


print("\n========================================")
print(" ON-TIME VS LATE ORDERS")
print("========================================")

print(
    delivery_status
)


plt.figure(figsize=(8, 6))

plt.bar(
    delivery_status.index,
    delivery_status.values
)

plt.title(
    "On-Time vs Late Deliveries"
)

plt.xlabel(
    "Delivery Status"
)

plt.ylabel(
    "Number of Orders"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/on_time_vs_late.png",
    dpi=300
)

plt.close()


# ==========================================
# 6. DELIVERY TIME BY STATUS
# ==========================================

on_time_days = delivered[
    delivered["late_delivery"] == 0
]["delivery_days"]

late_days = delivered[
    delivered["late_delivery"] == 1
]["delivery_days"]


print("\nAverage On-Time Delivery:")

print(
    round(
        on_time_days.mean(),
        2
    ),
    "days"
)


print(
    "Average Late Delivery:"
)

print(
    round(
        late_days.mean(),
        2
    ),
    "days"
)


plt.figure(figsize=(8, 6))

plt.boxplot(
    [
        on_time_days,
        late_days
    ],
    tick_labels=[
        "On-Time",
        "Late"
    ]
)

plt.title(
    "Delivery Time: On-Time vs Late Orders"
)

plt.ylabel(
    "Delivery Time (Days)"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/delivery_time_comparison.png",
    dpi=300
)

plt.close()
# ==========================================
# ADVANCED ANALYSIS
# ==========================================

print("\n========================================")
print(" ADVANCED LOGISTICS ANALYSIS")
print("========================================")


# ==========================================
# STATE DELIVERY PERFORMANCE
# ==========================================

customer_state = customers[
    ["customer_id", "customer_state"]
].drop_duplicates()

state_data = delivered.merge(
    customer_state,
    on="customer_id",
    how="left"
)

state_analysis = state_data.groupby(
    "customer_state"
).agg(
    total_orders=("order_id", "count"),
    average_delivery_days=("delivery_days", "mean"),
    late_rate=("late_delivery", "mean")
)

state_analysis["late_rate"] *= 100

print("\nSTATE DELIVERY PERFORMANCE")
print(state_analysis.round(2))


# ==========================================
# HIGH-RISK STATES
# ==========================================

high_risk_states = state_analysis[
    state_analysis["total_orders"] >= 100
].sort_values(
    "late_rate",
    ascending=False
).head(10)

print("\nHIGH-RISK STATES")
print(high_risk_states.round(2))


# ==========================================
# SELLER ANALYSIS
# ==========================================

seller_data = delivered[
    ["order_id", "delivery_days", "late_delivery"]
].merge(
    order_items[
        [
            "order_id",
            "seller_id",
            "price",
            "freight_value"
        ]
    ],
    on="order_id",
    how="inner"
)

seller_analysis = seller_data.groupby(
    "seller_id"
).agg(
    total_orders=("order_id", "count"),
    average_delivery_days=("delivery_days", "mean"),
    late_rate=("late_delivery", "mean"),
    average_price=("price", "mean"),
    average_freight=("freight_value", "mean")
)

seller_analysis["late_rate"] *= 100

reliable_sellers = seller_analysis[
    seller_analysis["total_orders"] >= 20
].sort_values(
    "late_rate",
    ascending=False
)

print("\nHIGH-RISK SELLERS")
print(
    reliable_sellers.head(10).round(2)
)


# ==========================================
# CORRELATION ANALYSIS
# ==========================================

delivery_items = delivered[
    [
        "order_id",
        "delivery_days",
        "late_delivery"
    ]
].merge(
    order_items[
        [
            "order_id",
            "price",
            "freight_value"
        ]
    ],
    on="order_id",
    how="inner"
)

correlation_matrix = delivery_items[
    [
        "delivery_days",
        "late_delivery",
        "price",
        "freight_value"
    ]
].corr()

print("\nCORRELATION MATRIX")
print(
    correlation_matrix.round(3)
)


# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title(
    "Logistics Feature Correlation Heatmap"
)

for i in range(
    len(correlation_matrix.columns)
):
    for j in range(
        len(correlation_matrix.columns)
    ):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "week3_charts/correlation_heatmap.png",
    dpi=300
)

plt.close()


# ==========================================
# FREIGHT VS DELIVERY TIME
# ==========================================

sample_delivery = delivery_items.sample(
    min(10000, len(delivery_items)),
    random_state=42
)

plt.figure(figsize=(10, 6))

plt.scatter(
    sample_delivery["freight_value"],
    sample_delivery["delivery_days"],
    alpha=0.4
)

plt.title(
    "Freight Value vs Delivery Time"
)

plt.xlabel(
    "Freight Value"
)

plt.ylabel(
    "Delivery Time (Days)"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/freight_vs_delivery_time.png",
    dpi=300
)

plt.close()




# ==========================================
# ON-TIME VS LATE
# ==========================================

delivery_status = (
    delivered["late_delivery"]
    .map({
        0: "On-Time",
        1: "Late"
    })
    .value_counts()
)

print("\nON-TIME VS LATE ORDERS")
print(delivery_status)

plt.figure(figsize=(8, 6))

plt.bar(
    delivery_status.index,
    delivery_status.values
)

plt.title(
    "On-Time vs Late Deliveries"
)

plt.xlabel(
    "Delivery Status"
)

plt.ylabel(
    "Number of Orders"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/on_time_vs_late.png",
    dpi=300
)

plt.close()


# ==========================================
# ON-TIME VS LATE DELIVERY TIME
# ==========================================

on_time_days = delivered[
    delivered["late_delivery"] == 0
]["delivery_days"].dropna()

late_days = delivered[
    delivered["late_delivery"] == 1
]["delivery_days"].dropna()

print(
    "\nAverage On-Time Delivery:",
    round(on_time_days.mean(), 2),
    "days"
)

print(
    "Average Late Delivery:",
    round(late_days.mean(), 2),
    "days"
)

plt.figure(figsize=(8, 6))

plt.boxplot(
    [
        on_time_days,
        late_days
    ],
    tick_labels=[
        "On-Time",
        "Late"
    ]
)

plt.title(
    "Delivery Time: On-Time vs Late Orders"
)

plt.ylabel(
    "Delivery Time (Days)"
)

plt.tight_layout()

plt.savefig(
    "week3_charts/delivery_time_comparison.png",
    dpi=300
)

plt.close()


# ==========================================
# STATE CHART
# ==========================================

plt.figure(figsize=(12, 7))

plot_states = state_analysis[
    state_analysis["total_orders"] >= 100
].sort_values(
    "late_rate",
    ascending=False
).head(10)

plt.bar(
    plot_states.index,
    plot_states["late_rate"]
)

plt.title(
    "Top 10 States by Late Delivery Rate"
)

plt.xlabel("State")

plt.ylabel(
    "Late Delivery Rate (%)"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "week3_charts/state_late_delivery_rate.png",
    dpi=300
)

plt.close()


# ==========================================
# SELLER CHART
# ==========================================

plt.figure(figsize=(12, 7))

top_sellers = reliable_sellers.head(10)

plt.bar(
    range(len(top_sellers)),
    top_sellers["late_rate"]
)

plt.title(
    "High-Risk Sellers by Late Delivery Rate"
)

plt.xlabel("Seller")

plt.ylabel(
    "Late Delivery Rate (%)"
)

plt.xticks(
    range(len(top_sellers)),
    [
        seller[:8]
        for seller in top_sellers.index
    ],
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "week3_charts/high_risk_sellers.png",
    dpi=300
)

plt.close()


# ==========================================
# SAVE ANALYSIS TABLES
# ==========================================

state_analysis.to_csv(
    "week3_state_analysis.csv"
)

seller_analysis.to_csv(
    "week3_seller_analysis.csv"
)

monthly.to_csv(
    "week3_monthly_analysis.csv"
)

print("\nAdvanced charts and analysis saved successfully!")

# ==========================================
# DELIVERY TIME COMPARISON
# ==========================================

on_time_days = delivered[
    delivered["late_delivery"] == 0
]["delivery_days"].dropna()

late_days = delivered[
    delivered["late_delivery"] == 1
]["delivery_days"].dropna()

print("\n========================================")
print(" DELIVERY TIME COMPARISON")
print("========================================")

print(
    "Average On-Time Delivery:",
    round(on_time_days.mean(), 2),
    "days"
)

print(
    "Average Late Delivery:",
    round(late_days.mean(), 2),
    "days"
)

plt.figure(figsize=(8, 6))

plt.boxplot(
    [
        on_time_days,
        late_days
    ],
    tick_labels=[
        "On-Time",
        "Late"
    ]
)

plt.title(
    "Delivery Time: On-Time vs Late Orders"
)

plt.xlabel("Delivery Status")
plt.ylabel("Delivery Time (Days)")

plt.tight_layout()

plt.savefig(
    "week3_charts/delivery_time_comparison.png",
    dpi=300
)

plt.close()

print("Delivery comparison chart saved successfully!")
# ==========================================
# 7. FINAL CHART LIST
# ==========================================

print("\n========================================")
print(" WEEK 3 CHARTS COMPLETED")
print("========================================")

charts = [
    "delivery_time_distribution.png",
    "monthly_order_volume.png",
    "monthly_late_rate.png",
    "freight_distribution.png",
    "price_vs_freight.png",
    "weight_vs_freight.png",
    "state_late_delivery_rate.png",
    "high_risk_sellers.png",
    "correlation_heatmap.png",
    "freight_vs_delivery_time.png",
    "on_time_vs_late.png",
    "delivery_time_comparison.png"
]

for chart in charts:
    print(chart)

    
