import pandas as pd

# ==========================================
# WEEK 2 - DATA COLLECTION & PREPROCESSING
# ==========================================

DATA_PATH = "data/"

# Load datasets
orders = pd.read_csv(
    DATA_PATH + "olist_orders_dataset.csv"
)

order_items = pd.read_csv(
    DATA_PATH + "olist_order_items_dataset.csv"
)

customers = pd.read_csv(
    DATA_PATH + "olist_customers_dataset.csv"
)

sellers = pd.read_csv(
    DATA_PATH + "olist_sellers_dataset.csv"
)

products = pd.read_csv(
    DATA_PATH + "olist_products_dataset.csv"
)

print("========================================")
print("     WEEK 2 DATA COLLECTION")
print("========================================")

print("\nOrders shape:", orders.shape)
print("Order Items shape:", order_items.shape)
print("Customers shape:", customers.shape)
print("Sellers shape:", sellers.shape)
print("Products shape:", products.shape)


# ==========================================
# INITIAL DATA INSPECTION
# ==========================================

print("\n========================================")
print("     DATA TYPES - ORDERS")
print("========================================")

print(orders.dtypes)


print("\n========================================")
print("     FIRST 5 ORDERS")
print("========================================")

print(orders.head())


# ==========================================
# MISSING VALUE ANALYSIS
# ==========================================

print("\n========================================")
print("     MISSING VALUES - ORDERS")
print("========================================")

missing_values = orders.isnull().sum()

print(missing_values)


# Missing percentage
missing_percentage = (
    orders.isnull().mean() * 100
)

print("\nMissing Percentage:")

print(
    missing_percentage.round(2)
)


# ==========================================
# DUPLICATE ANALYSIS
# ==========================================

print("\n========================================")
print("     DUPLICATE ANALYSIS")
print("========================================")

print(
    "Duplicate orders:",
    orders.duplicated().sum()
)

print(
    "Duplicate order IDs:",
    orders["order_id"].duplicated().sum()
)


# ==========================================
# UNIQUE VALUES
# ==========================================

print("\n========================================")
print("     UNIQUE VALUES")
print("========================================")

print(
    "Unique orders:",
    orders["order_id"].nunique()
)

print(
    "Unique customers:",
    orders["customer_id"].nunique()
)

print(
    "Order statuses:",
    orders["order_status"].unique()
)


# ==========================================
# BASIC NUMERICAL INSPECTION
# ==========================================

print("\n========================================")
print("     NUMERICAL SUMMARY")
print("========================================")

print(
    order_items[
        [
            "price",
            "freight_value"
        ]
    ].describe()
)


print("\n========================================")
print("     PRODUCT SUMMARY")
print("========================================")

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
# WEEK 2 - STEP 2
# MISSING VALUE TREATMENT
# ==========================================

print("\n========================================")
print("     MISSING VALUE TREATMENT")
print("========================================")


# Make copies so the original datasets remain unchanged
orders_clean = orders.copy()
order_items_clean = order_items.copy()
products_clean = products.copy()


# ==========================================
# 1. CONVERT DATE COLUMNS
# ==========================================

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders_clean[column] = pd.to_datetime(
        orders_clean[column],
        errors="coerce"
    )


print("\nDate columns converted successfully.")


# ==========================================
# 2. CHECK MISSING VALUES BEFORE TREATMENT
# ==========================================

print("\nMissing values BEFORE treatment:")

print(
    orders_clean.isnull().sum()
)


# ==========================================
# 3. HANDLE ORDER APPROVAL DATE
# ==========================================

# Missing approval dates are rare (0.16%).
# Keep them as NaT because artificial dates
# could introduce incorrect information.

print(
    "\nMissing approval dates retained:",
    orders_clean["order_approved_at"].isnull().sum()
)


# ==========================================
# 4. HANDLE CARRIER DELIVERY DATE
# ==========================================

# Missing carrier delivery dates may occur
# because an order was not shipped or the
# delivery process was incomplete.

print(
    "Missing carrier delivery dates retained:",
    orders_clean[
        "order_delivered_carrier_date"
    ].isnull().sum()
)


# ==========================================
# 5. HANDLE CUSTOMER DELIVERY DATE
# ==========================================

# Missing customer delivery dates are important.
# They may represent orders that were not delivered.

print(
    "Missing customer delivery dates retained:",
    orders_clean[
        "order_delivered_customer_date"
    ].isnull().sum()
)


# ==========================================
# 6. PRODUCT NUMERICAL MISSING VALUES
# ==========================================

product_numeric_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

print("\nProduct missing values BEFORE treatment:")

print(
    products_clean[
        product_numeric_columns
    ].isnull().sum()
)


# ==========================================
# 7. MEDIAN IMPUTATION FOR PRODUCT DATA
# ==========================================

for column in product_numeric_columns:

    median_value = products_clean[column].median()

    products_clean[column] = (
        products_clean[column]
        .fillna(median_value)
    )


print("\nProduct missing values AFTER treatment:")

print(
    products_clean[
        product_numeric_columns
    ].isnull().sum()
)


# ==========================================
# 8. VERIFY ORDER DATA
# ==========================================

print("\n========================================")
print("     FINAL ORDER MISSING VALUES")
print("========================================")

print(
    orders_clean.isnull().sum()
)


# ==========================================
# 9. SAVE CLEANED DATASETS
# ==========================================

orders_clean.to_csv(
    "week2_cleaned_orders.csv",
    index=False
)

order_items_clean.to_csv(
    "week2_cleaned_order_items.csv",
    index=False
)

products_clean.to_csv(
    "week2_cleaned_products.csv",
    index=False
)


print("\nCleaned datasets saved successfully!")

print(
    "week2_cleaned_orders.csv"
)

print(
    "week2_cleaned_order_items.csv"
)

print(
    "week2_cleaned_products.csv"
)
# ==========================================
# WEEK 2 - STEP 3
# DUPLICATE & INVALID VALUE CLEANING
# ==========================================

print("\n========================================")
print(" DUPLICATE & INVALID VALUE ANALYSIS")
print("========================================")


# ==========================================
# 1. DUPLICATE CHECK
# ==========================================

print("\nDuplicate rows:")

print(
    "Orders:",
    orders_clean.duplicated().sum()
)

print(
    "Order Items:",
    order_items_clean.duplicated().sum()
)

print(
    "Products:",
    products_clean.duplicated().sum()
)


# ==========================================
# 2. DUPLICATE ID CHECK
# ==========================================

print("\nDuplicate IDs:")

print(
    "Order IDs:",
    orders_clean["order_id"].duplicated().sum()
)

print(
    "Product IDs:",
    products_clean["product_id"].duplicated().sum()
)


# ==========================================
# 3. INVALID ORDER STATUS
# ==========================================

valid_statuses = [
    "delivered",
    "invoiced",
    "shipped",
    "processing",
    "unavailable",
    "canceled",
    "created",
    "approved"
]

invalid_status = orders_clean[
    ~orders_clean["order_status"].isin(valid_statuses)
]

print("\nInvalid order statuses:")

print(
    len(invalid_status)
)

if len(invalid_status) > 0:
    print(invalid_status["order_status"].unique())


# ==========================================
# 4. INVALID PRICE VALUES
# ==========================================

invalid_price = order_items_clean[
    order_items_clean["price"] <= 0
]

print("\nInvalid price records:")

print(
    len(invalid_price)
)


# ==========================================
# 5. INVALID FREIGHT VALUES
# ==========================================

invalid_freight = order_items_clean[
    order_items_clean["freight_value"] < 0
]

print("\nInvalid freight records:")

print(
    len(invalid_freight)
)


# ==========================================
# 6. INVALID PRODUCT WEIGHT
# ==========================================

invalid_weight = products_clean[
    products_clean["product_weight_g"] <= 0
]

print("\nInvalid product weight records:")

print(
    len(invalid_weight)
)


# ==========================================
# 7. INVALID PRODUCT DIMENSIONS
# ==========================================

invalid_length = products_clean[
    products_clean["product_length_cm"] <= 0
]

invalid_height = products_clean[
    products_clean["product_height_cm"] <= 0
]

invalid_width = products_clean[
    products_clean["product_width_cm"] <= 0
]

print("\nInvalid product dimensions:")

print(
    "Length:",
    len(invalid_length)
)

print(
    "Height:",
    len(invalid_height)
)

print(
    "Width:",
    len(invalid_width)
)


# ==========================================
# 8. CHECK ORDER DATE LOGIC
# ==========================================

invalid_purchase_dates = orders_clean[
    orders_clean["order_purchase_timestamp"].isna()
]

print("\nMissing purchase timestamps:")

print(
    len(invalid_purchase_dates)
)


# Approval before purchase
invalid_approval_dates = orders_clean[
    (
        orders_clean["order_approved_at"].notna()
    )
    &
    (
        orders_clean["order_approved_at"]
        < orders_clean["order_purchase_timestamp"]
    )
]

print("\nApproval before purchase:")

print(
    len(invalid_approval_dates)
)


# ==========================================
# 9. REMOVE DUPLICATES
# ==========================================

orders_clean = orders_clean.drop_duplicates()

order_items_clean = (
    order_items_clean.drop_duplicates()
)

products_clean = (
    products_clean.drop_duplicates()
)


print("\nDuplicate rows removed.")


# ==========================================
# 10. HANDLE INVALID NUMERICAL VALUES
# ==========================================

# Convert impossible values to missing values.
# They can then be handled using median imputation.

products_clean.loc[
    products_clean["product_weight_g"] <= 0,
    "product_weight_g"
] = pd.NA

products_clean.loc[
    products_clean["product_length_cm"] <= 0,
    "product_length_cm"
] = pd.NA

products_clean.loc[
    products_clean["product_height_cm"] <= 0,
    "product_height_cm"
] = pd.NA

products_clean.loc[
    products_clean["product_width_cm"] <= 0,
    "product_width_cm"
] = pd.NA


# Re-apply median imputation
product_numeric_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

for column in product_numeric_columns:

    products_clean[column] = (
        products_clean[column]
        .fillna(products_clean[column].median())
    )


# ==========================================
# 11. FINAL VALIDATION
# ==========================================

print("\n========================================")
print("       FINAL DATA QUALITY CHECK")
print("========================================")

print(
    "Orders:",
    orders_clean.shape
)

print(
    "Order Items:",
    order_items_clean.shape
)

print(
    "Products:",
    products_clean.shape
)

print(
    "\nRemaining duplicate orders:",
    orders_clean.duplicated().sum()
)

print(
    "Remaining duplicate products:",
    products_clean.duplicated().sum()
)

print(
    "\nRemaining product missing values:"
)

print(
    products_clean[
        product_numeric_columns
    ].isnull().sum()
)


# ==========================================
# 12. SAVE CLEANED DATA
# ==========================================

orders_clean.to_csv(
    "week2_final_orders.csv",
    index=False
)

order_items_clean.to_csv(
    "week2_final_order_items.csv",
    index=False
)

products_clean.to_csv(
    "week2_final_products.csv",
    index=False
)

print("\nFinal cleaned datasets saved successfully!")
# ==========================================
# WEEK 2 - STEP 4
# OUTLIER DETECTION USING IQR
# ==========================================

print("\n========================================")
print("       OUTLIER DETECTION")
print("========================================")


# ==========================================
# IQR FUNCTION
# ==========================================

def detect_outliers(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    outliers = df[
        (df[column] < lower_limit)
        |
        (df[column] > upper_limit)
    ]

    return {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit,
        "Outliers": len(outliers)
    }


# ==========================================
# 1. PRICE OUTLIERS
# ==========================================

price_result = detect_outliers(
    order_items_clean,
    "price"
)

print("\nPRICE OUTLIERS")

for key, value in price_result.items():

    print(
        key + ":",
        round(value, 2)
        if isinstance(value, float)
        else value
    )


# ==========================================
# 2. FREIGHT OUTLIERS
# ==========================================

freight_result = detect_outliers(
    order_items_clean,
    "freight_value"
)

print("\nFREIGHT VALUE OUTLIERS")

for key, value in freight_result.items():

    print(
        key + ":",
        round(value, 2)
        if isinstance(value, float)
        else value
    )


# ==========================================
# 3. PRODUCT WEIGHT OUTLIERS
# ==========================================

weight_result = detect_outliers(
    products_clean,
    "product_weight_g"
)

print("\nPRODUCT WEIGHT OUTLIERS")

for key, value in weight_result.items():

    print(
        key + ":",
        round(value, 2)
        if isinstance(value, float)
        else value
    )


# ==========================================
# 4. PRODUCT LENGTH OUTLIERS
# ==========================================

length_result = detect_outliers(
    products_clean,
    "product_length_cm"
)

print("\nPRODUCT LENGTH OUTLIERS")

for key, value in length_result.items():

    print(
        key + ":",
        round(value, 2)
        if isinstance(value, float)
        else value
    )


# ==========================================
# 5. PRODUCT HEIGHT OUTLIERS
# ==========================================

height_result = detect_outliers(
    products_clean,
    "product_height_cm"
)

print("\nPRODUCT HEIGHT OUTLIERS")

for key, value in height_result.items():

    print(
        key + ":",
        round(value, 2)
        if isinstance(value, float)
        else value
    )


# ==========================================
# 6. PRODUCT WIDTH OUTLIERS
# ==========================================

width_result = detect_outliers(
    products_clean,
    "product_width_cm"
)

print("\nPRODUCT WIDTH OUTLIERS")

for key, value in width_result.items():

    print(
        key + ":",
        round(value, 2)
        if isinstance(value, float)
        else value
    )


# ==========================================
# 7. OUTLIER SUMMARY TABLE
# ==========================================

outlier_summary = pd.DataFrame({
    "Variable": [
        "Price",
        "Freight Value",
        "Product Weight",
        "Product Length",
        "Product Height",
        "Product Width"
    ],

    "Outliers": [
        price_result["Outliers"],
        freight_result["Outliers"],
        weight_result["Outliers"],
        length_result["Outliers"],
        height_result["Outliers"],
        width_result["Outliers"]
    ]
})


print("\n========================================")
print("       OUTLIER SUMMARY")
print("========================================")

print(
    outlier_summary.to_string(index=False)
)
# ==========================================
# WEEK 2 - STEP 5
# OUTLIER TREATMENT + NORMALIZATION
# ==========================================

from sklearn.preprocessing import StandardScaler, MinMaxScaler

print("\n========================================")
print("       OUTLIER TREATMENT")
print("========================================")


# ==========================================
# IQR CAPPING FUNCTION
# ==========================================

def cap_outliers(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    df[column] = df[column].clip(
        lower=lower_limit,
        upper=upper_limit
    )

    return lower_limit, upper_limit


# ==========================================
# COPY DATA
# ==========================================

processed_items = order_items_clean.copy()
processed_products = products_clean.copy()


# ==========================================
# APPLY OUTLIER CAPPING
# ==========================================

item_columns = [
    "price",
    "freight_value"
]

product_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]


print("\nApplying IQR capping...")


for column in item_columns:

    lower, upper = cap_outliers(
        processed_items,
        column
    )

    print(
        column,
        "→ Lower:",
        round(lower, 2),
        "Upper:",
        round(upper, 2)
    )


for column in product_columns:

    lower, upper = cap_outliers(
        processed_products,
        column
    )

    print(
        column,
        "→ Lower:",
        round(lower, 2),
        "Upper:",
        round(upper, 2)
    )


print("\nOutlier treatment completed.")


# ==========================================
# NORMALIZATION / STANDARDIZATION
# ==========================================

print("\n========================================")
print("       FEATURE SCALING")
print("========================================")


# Select numerical logistics features

scaling_columns = [
    "price",
    "freight_value"
]


# Keep original values for comparison

original_values = processed_items[
    scaling_columns
].copy()


# ==========================================
# STANDARD SCALER
# ==========================================

standard_scaler = StandardScaler()

standard_scaled = standard_scaler.fit_transform(
    processed_items[scaling_columns]
)

standard_scaled_df = pd.DataFrame(
    standard_scaled,
    columns=[
        "price_standardized",
        "freight_standardized"
    ]
)


print("\nStandardized values:")

print(
    standard_scaled_df.head()
)


# ==========================================
# MIN-MAX SCALER
# ==========================================

minmax_scaler = MinMaxScaler()

minmax_scaled = minmax_scaler.fit_transform(
    processed_items[scaling_columns]
)

minmax_scaled_df = pd.DataFrame(
    minmax_scaled,
    columns=[
        "price_normalized",
        "freight_normalized"
    ]
)


print("\nMin-Max normalized values:")

print(
    minmax_scaled_df.head()
)


# ==========================================
# COMBINE SCALED FEATURES
# ==========================================

processed_items = pd.concat(
    [
        processed_items.reset_index(drop=True),
        standard_scaled_df,
        minmax_scaled_df
    ],
    axis=1
)


# ==========================================
# CHECK SCALING
# ==========================================

print("\n========================================")
print("       SCALING VALIDATION")
print("========================================")


print("\nStandardized feature statistics:")

print(
    processed_items[
        [
            "price_standardized",
            "freight_standardized"
        ]
    ].describe().round(2)
)


print("\nMin-Max feature statistics:")

print(
    processed_items[
        [
            "price_normalized",
            "freight_normalized"
        ]
    ].describe().round(2)
)


# ==========================================
# SAVE FINAL PROCESSED DATA
# ==========================================

processed_items.to_csv(
    "week2_processed_order_items.csv",
    index=False
)

processed_products.to_csv(
    "week2_processed_products.csv",
    index=False
)


print("\n========================================")
print(" FINAL PREPROCESSING COMPLETED")
print("========================================")

print(
    "Processed order items:",
    processed_items.shape
)

print(
    "Processed products:",
    processed_products.shape
)

print("\nFiles saved:")
print("week2_processed_order_items.csv")
print("week2_processed_products.csv")
# ==========================================
# WEEK 2 - STEP 6
# FINAL DATA QUALITY VALIDATION
# ==========================================

print("\n========================================")
print("     FINAL DATA QUALITY VALIDATION")
print("========================================")


# ==========================================
# 1. DATASET SHAPES
# ==========================================

print("\nDataset Shapes:")

print(
    "Orders:",
    orders_clean.shape
)

print(
    "Order Items:",
    processed_items.shape
)

print(
    "Products:",
    processed_products.shape
)


# ==========================================
# 2. DUPLICATE CHECK
# ==========================================

print("\n========================================")
print("     DUPLICATE VALIDATION")
print("========================================")

print(
    "Duplicate orders:",
    orders_clean.duplicated().sum()
)

print(
    "Duplicate order items:",
    processed_items.duplicated().sum()
)

print(
    "Duplicate products:",
    processed_products.duplicated().sum()
)


# ==========================================
# 3. MISSING VALUE CHECK
# ==========================================

print("\n========================================")
print("     MISSING VALUE VALIDATION")
print("========================================")

print("\nOrders:")

print(
    orders_clean.isnull().sum()
)

print("\nOrder Items:")

print(
    processed_items.isnull().sum()
)

print("\nProducts:")

print(
    processed_products.isnull().sum()
)


# ==========================================
# 4. INVALID NUMERICAL VALUES
# ==========================================

print("\n========================================")
print("     INVALID VALUE VALIDATION")
print("========================================")


print(
    "Negative prices:",
    (
        processed_items["price"] < 0
    ).sum()
)

print(
    "Negative freight:",
    (
        processed_items["freight_value"] < 0
    ).sum()
)

print(
    "Non-positive product weight:",
    (
        processed_products["product_weight_g"] <= 0
    ).sum()
)

print(
    "Non-positive product length:",
    (
        processed_products["product_length_cm"] <= 0
    ).sum()
)

print(
    "Non-positive product height:",
    (
        processed_products["product_height_cm"] <= 0
    ).sum()
)

print(
    "Non-positive product width:",
    (
        processed_products["product_width_cm"] <= 0
    ).sum()
)


# ==========================================
# 5. STANDARDIZATION VALIDATION
# ==========================================

print("\n========================================")
print("     STANDARDIZATION VALIDATION")
print("========================================")

standard_columns = [
    "price_standardized",
    "freight_standardized"
]

print(
    processed_items[
        standard_columns
    ].agg(
        ["mean", "std", "min", "max"]
    ).round(3)
)


# ==========================================
# 6. MIN-MAX VALIDATION
# ==========================================

print("\n========================================")
print("     MIN-MAX VALIDATION")
print("========================================")

normalized_columns = [
    "price_normalized",
    "freight_normalized"
]

print(
    processed_items[
        normalized_columns
    ].agg(
        ["min", "max"]
    ).round(3)
)


# ==========================================
# 7. DATA TYPES
# ==========================================

print("\n========================================")
print("     FINAL DATA TYPES")
print("========================================")

print(
    orders_clean.dtypes
)


# ==========================================
# 8. FINAL QUALITY REPORT
# ==========================================

print("\n========================================")
print("     WEEK 2 QUALITY REPORT")
print("========================================")

print(
    "Orders rows:",
    len(orders_clean)
)

print(
    "Order item rows:",
    len(processed_items)
)

print(
    "Product rows:",
    len(processed_products)
)

print(
    "Duplicate order rows:",
    orders_clean.duplicated().sum()
)

print(
    "Duplicate order-item rows:",
    processed_items.duplicated().sum()
)

print(
    "Duplicate product rows:",
    processed_products.duplicated().sum()
)


# ==========================================
# 9. SAVE QUALITY REPORT
# ==========================================

quality_report = pd.DataFrame({
    "Dataset": [
        "Orders",
        "Order Items",
        "Products"
    ],

    "Rows": [
        len(orders_clean),
        len(processed_items),
        len(processed_products)
    ],

    "Columns": [
        orders_clean.shape[1],
        processed_items.shape[1],
        processed_products.shape[1]
    ],

    "Duplicate Rows": [
        orders_clean.duplicated().sum(),
        processed_items.duplicated().sum(),
        processed_products.duplicated().sum()
    ]
})


quality_report.to_csv(
    "week2_quality_report.csv",
    index=False
)


print("\nQuality report saved as:")
print("week2_quality_report.csv")


print("\n========================================")
print("   WEEK 2 PREPROCESSING COMPLETED")
print("========================================")