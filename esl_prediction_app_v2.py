import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
import hashlib
import json
from datetime import datetime

# ========== 版本管理相关 ==========
DATA_PATH = 'DSL Data in Cincinnati.xlsx'
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
VERSION_RECORD = MODEL_DIR / 'model_versions.json'

REGION_COLUMNS = [
    'Central Market 1.54"', 'Central Market 2.66"',
    'Checkout 1.54"', 'Checkout 2.66"',
    'Pharmacy 1.54"', 'Pharmacy 2.66"',
    'Beer 1.54"', 'Beer 2.66"',
    'Cosmetics 1.54"',
    'Dairy 1.54"', 'Dairy 2.66"',
    'Seafood & Meat 1.54"', 'Seafood & Meat 2.66"', 'Seafood & Meat 4.20"',
    'Produce 1.54"', 'Produce 2.66"', 'Produce 4.20"',
    'Bakery 1.54"', 'Bakery 2.66"'
]
SIZE_COLUMNS = ['1.54"', '2.66"', '4.20"']
ESL_COLUMNS = REGION_COLUMNS + SIZE_COLUMNS

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def load_and_preprocess_data():
    df = pd.read_excel(DATA_PATH, sheet_name='工作表1', header=1)
    df.columns = df.columns.astype(str).str.strip()
    for col in ESL_COLUMNS + ['Total', 'Sales SQ FT']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Total ESL'] = df['Total']
    return df

def train_model(df):
    df_model = pd.get_dummies(df, columns=['Major Type'])
    X = df_model[['Sales SQ FT'] + [col for col in df_model.columns if col.startswith('Major Type_')]]
    y = df_model['Total ESL']
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

def save_model(model, version_hash):
    model_path = MODEL_DIR / f'esl_model_{version_hash}.joblib'
    joblib.dump(model, model_path)
    return str(model_path)

def load_version_record():
    if VERSION_RECORD.exists():
        with open(VERSION_RECORD, 'r') as f:
            return json.load(f)
    return {}

def save_version_record(record):
    with open(VERSION_RECORD, 'w') as f:
        json.dump(record, f, indent=2)

def get_latest_model():
    record = load_version_record()
    if not record:
        return None, None
    latest = sorted(record.items(), key=lambda x: x[1]['created'], reverse=True)[0]
    model_path = latest[1]['model_path']
    model = joblib.load(model_path)
    return model, latest[0]

def ensure_latest_model():
    data_hash = get_file_hash(DATA_PATH)
    record = load_version_record()
    if data_hash in record:
        model_path = record[data_hash]['model_path']
    else:
        df = load_and_preprocess_data()
        model = train_model(df)
        model_path = save_model(model, data_hash)
        record[data_hash] = {
            'model_path': model_path,
            'created': datetime.now().isoformat()
        }
        save_version_record(record)
    model = joblib.load(model_path)
    return model, data_hash

# ========== Streamlit 前端 ==========

def predict_new_store(model, sales_sqft, major_type, df):
    input_data = pd.DataFrame({
        'Sales SQ FT': [sales_sqft]
    })
    for col in df['Major Type'].unique():
        input_data[f'Major Type_{col}'] = 0
    input_data[f'Major Type_{major_type}'] = 1
    prediction = model.predict(input_data)[0]
    return round(prediction)

def predict_esl_distribution(sales_sqft, major_type, df, esl_columns, model):
    region_columns = [col for col in esl_columns if col not in ['1.54"', '2.66"', '4.20"']]
    avg_distribution = df[df['Major Type'] == major_type][region_columns].mean()
    total_esl = predict_new_store(model, sales_sqft, major_type, df)
    distribution = {}
    for col in region_columns:
        percentage = avg_distribution[col] / avg_distribution.sum()
        distribution[col] = round(total_esl * percentage)
    distribution['1.54"'] = sum(distribution[col] for col in region_columns if '1.54"' in col)
    distribution['2.66"'] = sum(distribution[col] for col in region_columns if '2.66"' in col)
    distribution['4.20"'] = sum(distribution[col] for col in region_columns if '4.20"' in col)
    return distribution

def main():
    st.set_page_config(
        page_title="ESL Prediction Tool",
        page_icon="🏪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
        .main {padding: 2rem;}
        .stButton>button {width: 100%;margin-top: 1rem;}
        .reportview-container {background: #f0f2f6;}
        .css-1d391kg {padding: 2rem 1rem;}
        h1 {color: #1f77b4;}
        </style>
        """, unsafe_allow_html=True)

    st.title("🏪 ESL Prediction System (with Model Versioning)")
    st.markdown("---")

    # 1. 数据和模型版本管理
    model, data_hash = ensure_latest_model()
    df = load_and_preprocess_data()
    esl_columns = ESL_COLUMNS

    with st.sidebar:
        st.header("Input Parameters")
        available_types = sorted(df['Major Type'].unique())
        sales_sqft = st.number_input(
            "Sales Area (SQ FT)",
            min_value=1000.0,
            max_value=200000.0,
            value=50000.0,
            step=1000.0,
            help="Enter the sales area in square feet"
        )
        selected_type = st.selectbox(
            "Store Type",
            options=available_types,
            help="Select the type of store"
        )
        predict_button = st.button("Generate Predictions", type="primary")

    if predict_button:
        with st.spinner('Generating predictions...'):
            total_prediction = predict_new_store(model, sales_sqft, selected_type, df)
            region_prediction = predict_esl_distribution(sales_sqft, selected_type, df, esl_columns, model)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📊 Prediction Results")
            st.metric(
                label="Total Predicted ESL Count",
                value=f"{total_prediction:,}",
                delta=f"{total_prediction/sales_sqft:.2f} per sq ft"
            )
            if region_prediction:
                st.subheader("📍 ESL Distribution by Region")
                total_keys = ['1.54"', '2.66"', '4.20"']
                region_keys = [k for k in region_prediction.keys() if k not in total_keys]
                display_keys = total_keys + region_keys
                df_results = pd.DataFrame({
                    'Region': display_keys,
                    'Count': [region_prediction[k] for k in display_keys]
                })
                df_results['Percentage'] = (df_results['Count'] / total_prediction * 100).round(2)
                df_results['Percentage'] = df_results['Percentage'].astype(str) + '%'
                st.dataframe(
                    df_results,
                    column_config={
                        "Region": "Region",
                        "Count": st.column_config.NumberColumn("Count", format="%d"),
                        "Percentage": "Percentage"
                    },
                    hide_index=True
                )
        with col2:
            st.subheader("📈 Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            regions = display_keys
            counts = [region_prediction[k] for k in display_keys]
            bars = ax.bar(regions, counts, color=sns.color_palette("viridis", len(regions)))
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.title(f"ESL Distribution - {selected_type}\n{sales_sqft:,.0f} SQ FT", pad=20)
            plt.xlabel("Region")
            plt.ylabel("ESL Count")
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom')
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("📊 Additional Analysis")
        st.write("Compare your store's predictions with typical distributions")
        df_region = df.groupby('Major Type')[esl_columns].mean()
        fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
        x = np.arange(len(display_keys))
        width = 0.35
        predicted_percentages = [region_prediction[k]/total_prediction*100 for k in display_keys]
        actual_counts = df_region.loc[selected_type][display_keys].values
        region_keys_only = [k for k in display_keys if k not in total_keys]
        region_sum = df_region.loc[selected_type][region_keys_only].sum()
        actual_percentages = []
        for k in display_keys:
            val = df_region.loc[selected_type][k] / region_sum * 100
            actual_percentages.append(val)
        ax_comp.bar(x - width/2, predicted_percentages, width, label='Predicted', 
                   color='skyblue')
        ax_comp.bar(x + width/2, actual_percentages, width, label='Average', 
                   color='lightgreen')
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(display_keys, rotation=45, ha='right')
        ax_comp.legend()
        ax_comp.set_ylabel('Percentage (%)')
        ax_comp.set_title('Predicted vs Average Distribution')
        plt.tight_layout()
        st.pyplot(fig_comp)
    else:
        st.info("""
        👋 Welcome to the ESL Prediction System!
        This tool helps you predict Electronic Shelf Label (ESL) requirements for your store. To get started:
        1. Enter your store's sales area in square feet
        2. Select your store type from the dropdown
        3. Click "Generate Predictions" to see detailed results
        The system will provide:
        - Total ESL count prediction
        - Detailed distribution by region
        - Visual analysis and comparisons
        """)

if __name__ == "__main__":
    main() 