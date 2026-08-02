import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration

st.set_page_config(
    page_title="Real Estate Buyer Segmentation Dashboard",
    layout="wide"
)

st.title("🏠 Machine Learning Based Buyer Segmentation & Investment Profiling")

# Load Dataset

data = pd.read_csv("Real_Estate_Dashboard.csv")


# Sidebar Filters


st.sidebar.header("Dashboard Filters")

country = st.sidebar.selectbox(
    "Country",
    ["All"] + sorted(data["country"].unique().tolist())
)

if country != "All":
    data = data[data["country"] == country]


region = st.sidebar.selectbox(
    "Region",
    ["All"] + sorted(data["region"].unique().tolist())
)

if region != "All":
    data = data[data["region"] == region]


client = st.sidebar.selectbox(
    "Client Type",
    ["All"] + sorted(data["client_type"].astype(str).unique().tolist())
)

if client != "All":
    data = data[data["client_type"] == client]

purpose = st.sidebar.selectbox(
    "Acquisition Purpose",
    ["All"] + sorted(data["acquisition_purpose"].unique().tolist())
)

if purpose != "All":
    data = data[data["acquisition_purpose"] == purpose]    

# KPI Cards

total_buyers = len(data)

total_sales = data["sale_price"].sum()

avg_satisfaction = data["satisfaction_score"].mean()

avg_sale = data["sale_price"].mean()



properties = len(data)

col1,col2,col3,col4,col5 = st.columns(5)

with col1:
    st.metric("👥 Total Buyers", total_buyers)

with col2:
    st.metric("💰 Total Sales Value", f"${total_sales:,.0f}")

with col3:
    st.metric("🏢 Properties Sold", properties)

with col4:
    st.metric("⭐ Avg Satisfaction", f"{avg_satisfaction:.2f}")
with col5:
    st.metric(
    "🏡 Avg Property Price",f"${avg_sale:,.0f}"
)

# Buyer Cluster Distribution

st.subheader("Buyer Cluster Distribution")

cluster_counts = (
    data["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_names = {
    0: "Premium Buyers",
    1: "First-Time Buyers",
    2: "Corporate Buyers",
    3: "Luxury Investors"
}

cluster_counts.index = [
    cluster_names[i] for i in cluster_counts.index
]

fig, ax = plt.subplots(figsize=(7,4))

cluster_counts.plot(
    kind="bar",
    color="steelblue",
    ax=ax
)

ax.set_title("Buyer Cluster Distribution")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Buyers")

st.pyplot(fig)    


# Buyer Type Distribution


st.subheader("Buyer Type Distribution")

client_counts = data["client_type"].value_counts()

fig, ax = plt.subplots(figsize=(6,4))

client_counts.plot(
    kind="bar",
    color=["#4CAF50", "#2196F3"],
    ax=ax
)

ax.set_title("Buyer Type Distribution")
ax.set_xlabel("Client Type")
ax.set_ylabel("Number of Buyers")
plt.xticks(rotation=0)

st.pyplot(fig)


# Acquisition Purpose Analysis

st.subheader("Acquisition Purpose Analysis")

purpose = data["acquisition_purpose"].value_counts()

fig, ax = plt.subplots(figsize=(6,4))

purpose.plot(
    kind="bar",
    color="orange",
    ax=ax
)

ax.set_title("Acquisition Purpose")
ax.set_xlabel("Purpose")
ax.set_ylabel("Number of Buyers")
plt.xticks(rotation=0)

st.pyplot(fig)


# Average Sale Price by Cluster


st.subheader("Average Sale Price by Cluster")

price_cluster = (
    data.groupby("Cluster")["sale_price"]
    .mean()
)

fig, ax = plt.subplots(figsize=(7,4))

price_cluster.plot(
    kind="bar",
    color="green",
    ax=ax
)

ax.set_title("Average Sale Price by Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Average Sale Price ($)")

st.pyplot(fig)


# Geographic Buyer Analysis


st.subheader("Geographic Buyer Analysis")

region_count = (
    data["region"]
    .value_counts()
    .head(10)
)

fig, ax = plt.subplots(figsize=(8,5))

region_count.plot(
    kind="bar",
    color="purple",
    ax=ax
)

ax.set_title("Top Regions by Number of Buyers")
ax.set_xlabel("Region")
ax.set_ylabel("Number of Buyers")

plt.xticks(rotation=45)

st.pyplot(fig)


# Loan Applied Analysis


st.subheader("Loan Applied Analysis")

loan = data["loan_applied"].value_counts()

fig, ax = plt.subplots(figsize=(6,4))

loan.plot(
    kind="bar",
    color=["#FF9800", "#03A9F4"],
    ax=ax
)

ax.set_title("Loan Applied")
ax.set_xlabel("Loan Applied")
ax.set_ylabel("Number of Buyers")

plt.xticks(rotation=0)

st.pyplot(fig)


# Referral Channel Analysis


st.subheader("Referral Channel Analysis")

referral = data["referral_channel"].value_counts()

fig, ax = plt.subplots(figsize=(8,5))

referral.plot(
    kind="bar",
    color="teal",
    ax=ax
)

ax.set_title("Referral Channel")
ax.set_xlabel("Channel")
ax.set_ylabel("Number of Buyers")

plt.xticks(rotation=45)

st.pyplot(fig)


# Satisfaction Score by Cluster


st.subheader("Average Satisfaction Score by Cluster")

satisfaction = (
    data.groupby("Cluster")["satisfaction_score"]
    .mean()
)

fig, ax = plt.subplots(figsize=(7,4))

satisfaction.plot(
    kind="bar",
    color="red",
    ax=ax
)

ax.set_title("Average Satisfaction Score")
ax.set_xlabel("Cluster")
ax.set_ylabel("Average Score")

st.pyplot(fig)


# Business Insights


st.subheader("Business Insights")

highest_cluster = (
    data.groupby("Cluster")["sale_price"]
    .mean()
    .idxmax()
)

highest_region = (
    data.groupby("region")["sale_price"]
    .mean()
    .idxmax()
)

highest_purpose = (
    data["acquisition_purpose"]
    .value_counts()
    .idxmax()
)

cluster_names = {
    0: "Premium Residential Buyers",
    1: "First-Time Buyers",
    2: "Corporate / Investment Buyers",
    3: "Luxury Investors"
}

st.success(
    f"🏆 Highest Value Buyer Segment: {cluster_names[highest_cluster]}"
)

st.info(f"🌍 Highest Average Property Price Region: {highest_region}")

st.warning(f"🏠 Most Common Acquisition Purpose: {highest_purpose}")

st.write(f"👥 Total Buyers Analysed: **{total_buyers:,}**")

st.write(f"💰 Total Sales Value: **${total_sales:,.2f}**")


# Dataset Preview


with st.expander("📄 View Dataset"):
    st.dataframe(data)


# Download Dataset


csv = data.to_csv(index=False)

st.download_button(
    label="📥 Download Filtered Dataset",
    data=csv,
    file_name="Filtered_Real_Estate_Data.csv",
    mime="text/csv"
)

st.markdown("---")

st.markdown(
"""
### Project Information

**Project:** Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

**Developed By:** Ayshath  Afeesa P A

**Tools Used:** Python | Pandas | Matplotlib | Scikit-learn | Streamlit

**Machine Learning Algorithm:** K-Means Clustering
"""
)

st.markdown("---")
st.caption("© 2026 | Data Analytics Internship Project")
