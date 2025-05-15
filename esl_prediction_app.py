import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="ESL Prediction Tool",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

def load_and_preprocess_data():
    """Load and preprocess the data"""
    try:
        # Read the Excel file
        df = pd.read_excel("DSL Data in Cincinnati.xlsx", sheet_name='工作表1', header=1)
        
        # Clean column names
        df.columns = df.columns.astype(str).str.strip()
        
        # Define ESL columns
        region_columns = [
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
        size_columns = ['1.54"', '2.66"', '4.20"']
        esl_columns = region_columns + size_columns
        
        # Convert specified columns to numeric values
        for col in esl_columns + ['Total', 'Sales SQ FT']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create Total ESL column
        df['Total ESL'] = df['Total']
        
        return df, esl_columns
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def train_and_save_model(df):
    """Train the model and save it to disk"""
    # Create dummy variables for 'Major Type'
    df_model = pd.get_dummies(df, columns=['Major Type'])
    
    # Define features and target
    X = df_model[['Sales SQ FT'] + [col for col in df_model.columns if col.startswith('Major Type_')]]
    y = df_model['Total ESL']
    
    # Train the model
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Save the model
    model_path = Path("models/esl_model.joblib")
    joblib.dump(reg, model_path)
    
    return reg

def load_model():
    """Load the trained model from disk"""
    model_path = Path("models/esl_model.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None

def predict_new_store(model, sales_sqft, major_type, df):
    """Predict total ESL count for a new store"""
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Sales SQ FT': [sales_sqft]
    })
    
    # Add dummy variables for Major Type
    for col in df['Major Type'].unique():
        input_data[f'Major Type_{col}'] = 0
    input_data[f'Major Type_{major_type}'] = 1
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return round(prediction)

def predict_esl_distribution(sales_sqft, major_type, df, esl_columns):
    """Predict ESL distribution by region and size"""
    # Only use region columns for distribution
    region_columns = [col for col in esl_columns if col not in ['1.54"', '2.66"', '4.20"']]
    avg_distribution = df[df['Major Type'] == major_type][region_columns].mean()
    model = load_model()
    if model is None:
        model = train_and_save_model(df)
    total_esl = predict_new_store(model, sales_sqft, major_type, df)
    # Calculate region distribution
    distribution = {}
    for col in region_columns:
        percentage = avg_distribution[col] / avg_distribution.sum()
        distribution[col] = round(total_esl * percentage)
    # Dynamically calculate size totals
    distribution['1.54"'] = sum(distribution[col] for col in region_columns if '1.54"' in col)
    distribution['2.66"'] = sum(distribution[col] for col in region_columns if '2.66"' in col)
    distribution['4.20"'] = sum(distribution[col] for col in region_columns if '4.20"' in col)
    return distribution

def main():
    # Header
    st.title("🏪 ESL Prediction System")
    st.markdown("---")

    # Load and preprocess data
    df, esl_columns = load_and_preprocess_data()
    if df is None:
        return

    # Sidebar
    with st.sidebar:
        st.header("Input Parameters")
        
        # Get available store types
        available_types = sorted(df['Major Type'].unique())

        # Input fields
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

    # Main content area
    if predict_button:
        # Show loading message while training/predicting
        with st.spinner('Training model and generating predictions...'):
            # Train or load model
            model = load_model()
            if model is None:
                model = train_and_save_model(df)
            
            # Get predictions
            total_prediction = predict_new_store(model, sales_sqft, selected_type, df)
            region_prediction = predict_esl_distribution(sales_sqft, selected_type, df, esl_columns)

        # Create two columns for results
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Prediction Results")
            
            # Display total prediction in a metric
            st.metric(
                label="Total Predicted ESL Count",
                value=f"{total_prediction:,}",
                delta=f"{total_prediction/sales_sqft:.2f} per sq ft"
            )

            if region_prediction:
                st.subheader("📍 ESL Distribution by Region")
                
                # Define the new order
                total_keys = ['1.54"', '2.66"', '4.20"']
                region_keys = [k for k in region_prediction.keys() if k not in total_keys]
                display_keys = total_keys + region_keys
                
                # Convert to DataFrame for better display
                df_results = pd.DataFrame({
                    'Region': display_keys,
                    'Count': [region_prediction[k] for k in display_keys]
                })
                df_results['Percentage'] = (df_results['Count'] / total_prediction * 100).round(2)
                df_results['Percentage'] = df_results['Percentage'].astype(str) + '%'
                
                # Display as a styled table
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
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            regions = display_keys
            counts = [region_prediction[k] for k in display_keys]
            
            # Create bar plot
            bars = ax.bar(regions, counts, color=sns.color_palette("viridis", len(regions)))
            
            # Customize plot
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.title(f"ESL Distribution - {selected_type}\n{sales_sqft:,.0f} SQ FT", pad=20)
            plt.xlabel("Region")
            plt.ylabel("ESL Count")
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Display plot in Streamlit
            st.pyplot(fig)

        # Additional Analysis Section
        st.markdown("---")
        st.subheader("📊 Additional Analysis")
        
        # Create comparison chart
        st.write("Compare your store's predictions with typical distributions")
        
        # Calculate average distribution from the dataset
        df_region = df.groupby('Major Type')[esl_columns].mean()
        
        # Create comparison bar chart
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
        # Display welcome message and instructions when no prediction is made yet
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