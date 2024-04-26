import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
import plotly.express as px

# Function to calculate RFM scores
def calculate_rfm(data):
    # Convert 'DateOfOrder' column to datetime
    data['DateOfOrder'] = pd.to_datetime(data['DateOfOrder'])

    # Calculate Recency, Frequency, Monetary
    current_date = data['DateOfOrder'].max()  # Assuming current date as the max date in the dataset
    rfm_data = data.groupby('CustomerID').agg({
        'DateOfOrder': lambda x: (current_date - x.max()).days,  # Recency
        'OrderNumber': 'count',  # Frequency
        'ValueOfOrder': 'sum'  # Monetary
    }).reset_index()

    # Rename columns
    rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    return rfm_data

# Function to perform RFM analysis and binning
def perform_rfm_and_binning(data):
    # Calculate RFM scores
    rfm_data = calculate_rfm(data)

    # Binning using 5 quantiles
    binning = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    rfm_scores = binning.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])
    rfm_data['RFM_Score'] = rfm_scores.sum(axis=1)  # Sum of RFM scores
    
    # Define categories based on RFM score
    bins_labels = ['Churn', 'Average', 'Loyal', 'Best']
    rfm_data['RFM_Category'] = pd.cut(rfm_data['RFM_Score'], bins=[-1, 1, 2, 3, 5], labels=bins_labels)
    
    return rfm_data

# Streamlit app layout
st.title('Customer Churn Prediction App')
st.write("Developed by Shubham Raj. Contact support at sr6760.sr@gmail.com")

# Add the image
st.image("https://www.touchpoint.com/wp-content/uploads/2023/02/5.-Customer-churn-article.png", caption="Source: Touchpoint", use_column_width=True)

# Instructions for the required Excel data
st.write("Please upload an Excel file containing the following columns:")
st.write("- CustomerID")
st.write("- Name")
st.write("- OrderNumber")
st.write("- DateOfOrder")
st.write("- ValueOfOrder")

# Upload Excel file
uploaded_file = st.file_uploader('Upload Excel file', type=['xlsx', 'xls'])

if uploaded_file is not None:
    # Load data from Excel file
    data = pd.read_excel(uploaded_file)

    # Perform RFM analysis and binning
    rfm_data = perform_rfm_and_binning(data)

    # Churn Rate Report
    st.subheader('Churn Rate Report')
    category_counts = rfm_data['RFM_Category'].value_counts()
    churn_rate = category_counts['Churn'] / len(rfm_data) * 100
    st.write(f"Churn Rate: {churn_rate:.2f}%")

    # Scatter Plot of RFM Scores
    st.subheader('Scatter Plot of RFM Scores')
    plt.scatter(rfm_data['Recency'], rfm_data['Frequency'], c=rfm_data['Monetary'], cmap='viridis')
    plt.xlabel('Recency')
    plt.ylabel('Frequency')
    plt.colorbar(label='Monetary')
    st.pyplot(plt)

    # Box Plot of RFM Scores by Category
    st.subheader('Box Plot of RFM Scores by Category')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='RFM_Category', y='Recency', data=rfm_data)
    plt.xlabel('RFM Category')
    plt.ylabel('Recency')
    st.pyplot(plt)

    # Histograms of Recency, Frequency, and Monetary
    st.subheader('Histograms of RFM Components')
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(rfm_data['Recency'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Recency')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 2)
    plt.hist(rfm_data['Frequency'], bins=20, color='salmon', edgecolor='black')
    plt.xlabel('Frequency')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 3)
    plt.hist(rfm_data['Monetary'], bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Monetary')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # Heatmap of RFM Scores
    st.subheader('Heatmap of RFM Scores')
    plt.figure(figsize=(8, 6))
    sns.heatmap(rfm_data[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

    # Pie Chart of RFM Categories
    st.subheader('Pie Chart of RFM Categories')
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(plt)

    # Generate Churn Report in Excel
    st.subheader('Download Churn Report')
    st.write("Download the churn report as an Excel file.")
    churn_report = rfm_data.groupby('RFM_Category')['CustomerID'].apply(list).reset_index()
    churn_report.columns = ['RFM_Category', 'CustomerIDs']
    st.write(churn_report)
    st.download_button(label="Download Churn Report", data=churn_report.to_csv(index=False), file_name="churn_report.csv", mime="text/csv")
