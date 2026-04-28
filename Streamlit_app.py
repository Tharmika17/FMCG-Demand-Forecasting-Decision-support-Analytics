import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="FMCG Forecasting Dashboard", layout="wide")

# Load files
@st.cache_data
def load_data():
    processed_df = pd.read_csv("processed_fmcg.csv")
    engineered_df = pd.read_csv("processed_fmcg_engineered.csv")
    evaluation_df = pd.read_csv("evaluation_predictions.csv")
    sku_df = pd.read_csv("sku_level_metrics.csv")
    category_df = pd.read_csv("category_level_metrics.csv")
    shap_df = pd.read_csv("shap_feature_importance.csv")

    if "date" in processed_df.columns:
        processed_df["date"] = pd.to_datetime(processed_df["date"])
    if "date" in engineered_df.columns:
        engineered_df["date"] = pd.to_datetime(engineered_df["date"])
    if "date" in evaluation_df.columns:
        evaluation_df["date"] = pd.to_datetime(evaluation_df["date"])

    return processed_df, engineered_df, evaluation_df, sku_df, category_df, shap_df


@st.cache_resource
def load_model():
    return joblib.load("final_lightgbm_model.pkl")


processed_df, engineered_df, evaluation_df, sku_df, category_df, shap_df = load_data()
model = load_model()

# Month order
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

if "date" in processed_df.columns:
    processed_df["month_name"] = processed_df["date"].dt.month_name()
    processed_df["month_name"] = pd.Categorical(
        processed_df["month_name"],
        categories=month_order,
        ordered=True
    )

if "date" in evaluation_df.columns:
    evaluation_df["month_name"] = evaluation_df["date"].dt.month_name()
    evaluation_df["month_name"] = pd.Categorical(
        evaluation_df["month_name"],
        categories=month_order,
        ordered=True
    )

# Sidebar
st.sidebar.title("📊 FMCG Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Sales & Demand Analysis",
        "Inventory & Stock Insights",
        "Forecasting"
    ]
)

selected_category = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(processed_df["category"].astype(str).unique().tolist())
)

selected_region = st.sidebar.selectbox(
    "Region",
    ["All"] + sorted(processed_df["region"].astype(str).unique().tolist())
)

selected_channel = st.sidebar.selectbox(
    "Channel",
    ["All"] + sorted(processed_df["channel"].astype(str).unique().tolist())
)

selected_promotion = st.sidebar.selectbox(
    "Promotion",
    ["All", 0, 1]
)

# Filter helpers
def apply_common_filters(df):
    filtered = df.copy()

    if selected_category != "All" and "category" in filtered.columns:
        filtered = filtered[filtered["category"].astype(str) == selected_category]

    if selected_region != "All" and "region" in filtered.columns:
        filtered = filtered[filtered["region"].astype(str) == selected_region]

    if selected_channel != "All" and "channel" in filtered.columns:
        filtered = filtered[filtered["channel"].astype(str) == selected_channel]

    if selected_promotion != "All" and "promotion_flag" in filtered.columns:
        filtered = filtered[filtered["promotion_flag"] == selected_promotion]

    return filtered


filtered_df = apply_common_filters(processed_df)
filtered_evaluation_df = apply_common_filters(evaluation_df)

if "units_sold" in filtered_df.columns and "price_unit" in filtered_df.columns:
    filtered_df["revenue"] = filtered_df["units_sold"] * filtered_df["price_unit"]

# Page 1: Executive overview
if page == "Executive Overview":
    st.title("📌 Executive Overview")

    total_units = filtered_df["units_sold"].sum()
    total_revenue = filtered_df["revenue"].sum()
    avg_monthly_sales = filtered_df.groupby("month_name")["units_sold"].sum().mean()
    avg_price = filtered_df["price_unit"].mean()
    total_records = len(filtered_df)
    promo_sales = filtered_df[filtered_df["promotion_flag"] == 1]["units_sold"].sum()
    promo_pct = (promo_sales / total_units) * 100 if total_units != 0 else 0

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric("Total Units Sold", f"{total_units:,.0f} units")
    c2.metric("Total Revenue", f"${total_revenue:,.2f}")
    c3.metric("Average Monthly Sales", f"{avg_monthly_sales:,.0f} units")
    c4.metric("Average Price per Unit", f"${avg_price:,.2f} / unit")
    c5.metric("Total Records", f"{total_records:,}")
    c6.metric("Promotion Contribution", f"{promo_pct:.2f}%")

    st.subheader("Monthly Units Sold")
    monthly_sales = filtered_df.groupby("month_name")["units_sold"].sum().reindex(month_order)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(month_order, monthly_sales.values)
    ax.set_xlabel("Month")
    ax.set_ylabel("Units Sold")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Monthly Revenue")
    monthly_revenue = filtered_df.groupby("month_name")["revenue"].sum().reindex(month_order)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(month_order, monthly_revenue.values)
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Categories by Total Sales")
        cat_sales = filtered_df.groupby("category")["units_sold"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cat_sales.index.astype(str), cat_sales.values)
        ax.set_ylabel("Units Sold")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Top SKUs by Total Sales")
        sku_sales = filtered_df.groupby("sku")["units_sold"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(sku_sales.index.astype(str), sku_sales.values)
        ax.set_ylabel("Units Sold")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Page 2: Sales and demand analysis
elif page == "Sales & Demand Analysis":
    st.title("📈 Sales & Demand Analysis")

    monthly_total = filtered_df.groupby("month_name")["units_sold"].sum().reindex(month_order)
    monthly_mean = filtered_df.groupby("month_name")["units_sold"].mean().reindex(month_order)

    peak_month = monthly_total.idxmax()
    peak_value = monthly_total.max()
    low_month = monthly_total.idxmin()
    low_value = monthly_total.min()
    mean_monthly = monthly_total.mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak Month", f"{peak_month} ({peak_value:,.0f} units)")
    c2.metric("Lowest Month", f"{low_month} ({low_value:,.0f} units)")
    c3.metric("Mean Monthly Sales", f"{mean_monthly:,.0f} units)")

    st.subheader("Monthly Total Units Sold")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(month_order, monthly_total.values)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Units Sold")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category-wise Average Demand")
        cat_mean = filtered_df.groupby("category")["units_sold"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cat_mean.index.astype(str), cat_mean.values)
        ax.set_ylabel("Average Units Sold")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Month-wise Mean Demand")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(month_order, monthly_mean.values)
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean Units Sold")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Page 3: Inventory and stock insights
elif page == "Inventory & Stock Insights":
    st.title("📦 Inventory & Stock Insights")

    inventory_df = filtered_df.copy()

    inventory_df["stock_utilization"] = (
        inventory_df["units_sold"] / inventory_df["stock_available"].replace(0, np.nan)
    )
    inventory_df["stock_utilization"] = inventory_df["stock_utilization"].fillna(0)

    inventory_df["stock_status"] = np.where(
        inventory_df["stock_available"] < inventory_df["units_sold"],
        "Stockout Risk",
        np.where(
            inventory_df["stock_available"] > inventory_df["units_sold"] * 1.5,
            "Overstock",
            "Optimal"
        )
    )

    total_stock = inventory_df["stock_available"].sum()
    avg_stock = inventory_df["stock_available"].mean()
    avg_util = inventory_df["stock_utilization"].mean()
    stockout_count = (inventory_df["stock_status"] == "Stockout Risk").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stock", f"{total_stock:,.0f} units")
    c2.metric("Average Stock", f"{avg_stock:,.0f} units")
    c3.metric("Average Stock Utilization", f"{avg_util:.2f}")
    c4.metric("Stockout Risk Count", f"{stockout_count:,}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Average Stock")
        monthly_stock = inventory_df.groupby("month_name")["stock_available"].mean().reindex(month_order)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(month_order, monthly_stock.values)
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Stock Available (units)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Average Stock Utilization")
        monthly_util = inventory_df.groupby("month_name")["stock_utilization"].mean().reindex(month_order)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(month_order, monthly_util.values)
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Stock Utilization")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Stock Available vs Units Sold")
    sample_df = inventory_df.sample(min(3000, len(inventory_df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample_df["stock_available"], sample_df["units_sold"], alpha=0.4)
    ax.set_xlabel("Stock Available (units)")
    ax.set_ylabel("Units Sold")
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stock Status Distribution")
        status_counts = inventory_df["stock_status"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%")
        st.pyplot(fig)

    with col2:
        st.subheader("Stockout Risk by Category")
        risk_cat = (
            inventory_df[inventory_df["stock_status"] == "Stockout Risk"]
            .groupby("category")
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(risk_cat.index.astype(str), risk_cat.values)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Inventory Table")
    st.dataframe(
        inventory_df[["date", "sku", "category", "stock_available", "units_sold", "stock_status"]].head(50),
        use_container_width=True
    )

# Page 4: Forecasting
elif page == "Forecasting":
    st.title("🔮 Forecasting")

    eval_df = filtered_evaluation_df.copy()

    if len(eval_df) == 0:
        st.warning("No forecasting data available for the selected sidebar filters.")
    else:
        mae = np.mean(np.abs(eval_df["units_sold"] - eval_df["predicted_units_sold"]))
        rmse = np.sqrt(np.mean((eval_df["units_sold"] - eval_df["predicted_units_sold"]) ** 2))

        ss_res = np.sum((eval_df["units_sold"] - eval_df["predicted_units_sold"]) ** 2)
        ss_tot = np.sum((eval_df["units_sold"] - eval_df["units_sold"].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        accuracy = 100 - (
            np.mean(
                np.abs(
                    (eval_df["units_sold"] - eval_df["predicted_units_sold"]) /
                    np.where(eval_df["units_sold"] == 0, 1, eval_df["units_sold"])
                )
            ) * 100
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("R²", f"{r2:.3f}")
        c4.metric("Forecast Accuracy", f"{accuracy:.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Actual vs Predicted Scatter")
            plot_df = eval_df.sample(min(3000, len(eval_df)), random_state=42)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(plot_df["units_sold"], plot_df["predicted_units_sold"], alpha=0.4)
            ax.set_xlabel("Actual Units Sold")
            ax.set_ylabel("Predicted Units Sold")
            st.pyplot(fig)

        with col2:
            st.subheader("Error Distribution")
            errors = eval_df["units_sold"] - eval_df["predicted_units_sold"]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(errors, bins=30)
            ax.set_xlabel("Prediction Error")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        st.subheader("Prediction Results")
        st.dataframe(eval_df.head(50), use_container_width=True)

    st.subheader("Feature Importance")
    feature_col = shap_df.columns[0]
    importance_col = shap_df.columns[1]

    top_shap = shap_df.sort_values(by=importance_col, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_shap[feature_col].astype(str)[::-1], top_shap[importance_col][::-1])
    ax.set_xlabel(importance_col)
    st.pyplot(fig)

    st.subheader("Model Performance by Category and SKU")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.markdown("**Category-Level Metrics**")
        st.dataframe(category_df.head(20), use_container_width=True)

        category_numeric_cols = category_df.select_dtypes(include=np.number).columns.tolist()

        if len(category_numeric_cols) > 0:
            category_metric = category_numeric_cols[0]

            if "category" in category_df.columns:
                category_label_col = "category"
            elif "Model" in category_df.columns:
                category_label_col = "Model"
            else:
                category_label_col = category_df.columns[0]

            plot_df = category_df[[category_label_col, category_metric]].head(10)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(plot_df[category_label_col].astype(str), plot_df[category_metric])
            ax.set_xlabel(category_label_col)
            ax.set_ylabel(category_metric)
            ax.set_title(f"{category_metric} by Category")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found in category-level metrics file.")

    with perf_col2:
        st.markdown("**SKU-Level Metrics**")
        st.dataframe(sku_df.head(20), use_container_width=True)

        sku_numeric_cols = sku_df.select_dtypes(include=np.number).columns.tolist()

        if len(sku_numeric_cols) > 0:
            sku_metric = sku_numeric_cols[0]

            if "sku" in sku_df.columns:
                sku_label_col = "sku"
            elif "Model" in sku_df.columns:
                sku_label_col = "Model"
            else:
                sku_label_col = sku_df.columns[0]

            top_sku = sku_df.sort_values(by=sku_metric, ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(top_sku[sku_label_col].astype(str), top_sku[sku_metric])
            ax.set_xlabel(sku_label_col)
            ax.set_ylabel(sku_metric)
            ax.set_title(f"{sku_metric} by SKU")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found in SKU-level metrics file.")

    st.subheader("Interactive Demand Prediction")

    prediction_base_df = engineered_df.copy()

    # Category
    category_options = sorted(engineered_df["category"].astype(str).unique().tolist())
    input_category = st.selectbox("Category", category_options)

    # SKU (depends on category)
    sku_options = sorted(
        engineered_df.loc[
            engineered_df["category"].astype(str) == input_category, "sku"
        ].astype(str).unique().tolist()
    )
    input_sku = st.selectbox("SKU", sku_options)

    # Region
    region_options = sorted(engineered_df["region"].astype(str).unique().tolist())
    input_region = st.selectbox("Region", region_options)

    # Channel
    channel_options = sorted(engineered_df["channel"].astype(str).unique().tolist())
    input_channel = st.selectbox("Channel", channel_options)

    # Other inputs
    input_price = st.number_input("Price per Unit ($)", min_value=0.0, value=10.0, step=0.1)
    input_promo = st.selectbox("Promotion Flag", [0, 1])
    input_stock = st.number_input("Stock Available (units)", min_value=0, value=100, step=1)


    # Prediction
    if st.button("Predict Demand"):
        base_candidates = prediction_base_df[
            (prediction_base_df["category"].astype(str) == str(input_category)) &
            (prediction_base_df["sku"].astype(str) == str(input_sku)) &
            (prediction_base_df["region"].astype(str) == str(input_region)) &
            (prediction_base_df["channel"].astype(str) == str(input_channel))
        ].copy()

        if base_candidates.empty:
            st.error("No matching historical engineered row found for selected inputs.")
        else:
            if "date" in base_candidates.columns:
                base_candidates = base_candidates.sort_values("date")

            base_row = base_candidates.iloc[-1].copy()

            # User input override
            base_row["category"] = input_category
            base_row["sku"] = input_sku
            base_row["region"] = input_region
            base_row["channel"] = input_channel
            base_row["price_unit"] = input_price
            base_row["promotion_flag"] = input_promo
            base_row["stock_available"] = input_stock


            # Interaction features
            if "promo_price_interaction" in base_row.index:
                base_row["promo_price_interaction"] = input_promo * input_price

            if "is_stockout" in base_row.index:
                base_row["is_stockout"] = 1 if input_stock <= 0 else 0


            # Final prediction

            pred_df = pd.DataFrame([base_row]).drop(columns=["units_sold"], errors="ignore")

            train_feature_cols = [col for col in engineered_df.columns if col != "units_sold"]
            pred_df = pred_df.reindex(columns=train_feature_cols)

            prediction = model.predict(pred_df)[0]
            predicted_units = max(0, round(prediction))

            st.success(f"Predicted Demand: {predicted_units} units")