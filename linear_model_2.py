import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# === 1. Read and Clean Data ===
file_path = "DSL Data in Cincinnati.xlsx"
# Read the Excel file; header row is the second row (index=1)
df = pd.read_excel(file_path, sheet_name='工作表1', header=1)
# Clean column names by stripping extra whitespace
df.columns = df.columns.astype(str).str.strip()

# === 2. Define ESL Region Columns ===
esl_columns = [
    '1.54"', '2.66"', '4.20"',
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

# Convert specified columns to numeric values (including 'Total' and 'Sales SQ FT')
for col in esl_columns + ['Total', 'Sales SQ FT']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create a column for Total ESL (using the 'Total' column)
df['Total ESL'] = df['Total']

# === 3. Calculate Percentage Proportions for Each ESL Region ===
for col in esl_columns:
    df[f"{col} %"] = df[col] / df['Total ESL']

# === 4. Group Statistics: Average ESL Distribution by Major Type ===
# This gives us an average percentage for each ESL region by Major Type
grouped_avg = df.groupby('Major Type')[[f"{col} %" for col in esl_columns]].mean()
# Save the grouped averages to an Excel file for reference
grouped_avg.to_excel("ESL_Grouped_Averages_by_Type.xlsx")
print("Grouped averages by Major Type saved to ESL_Grouped_Averages_by_Type.xlsx")

# === 5. Build a Linear Regression Model to Predict Total ESL Using Sales SQ FT and Major Type ===
# Filter out records with missing values in Sales SQ FT and Total ESL
df_model = df.copy()
df_model = df_model[df_model['Sales SQ FT'].notnull() & df_model['Total ESL'].notnull()]
# Create dummy variables for 'Major Type'
df_model = pd.get_dummies(df_model, columns=['Major Type'])
print(df_model.head())

# Define features: Sales SQ FT and dummy columns for Major Type
X = df_model[['Sales SQ FT'] + [col for col in df_model.columns if col.startswith('Major Type_')]]
y = df_model['Total ESL']

# Train the linear regression model
reg = LinearRegression()
reg.fit(X, y)

# === 6. Output Model Coefficients ===
print("✅ Linear Regression Model Coefficients:")
for name, coef in zip(X.columns, reg.coef_):
    print(f"{name}: {coef:.2f}")
print(f"Intercept: {reg.intercept_:.2f}")

# === 7. Define a Prediction Function for Total ESL of a New Store ===
def predict_new_store(sales_sqft, major_type=None):
    """
    Predict total ESL count for a new store given its Sales SQ FT and Major Type.
    
    Parameters:
    -----------
    sales_sqft : float
        Sales area in square feet
    major_type : str, optional
        Store type. If None, will use the most common type in the dataset
        
    Returns:
    --------
    int
        Predicted total ESL count
    """
    # Default to most common major type if none provided
    if major_type is None:
        major_type = df['Major Type'].value_counts().index[0]
        print(f"No store type specified. Using most common type: {major_type}")
    
    # Validate major_type exists in the model
    target_col = f'Major Type_{major_type.upper()}'
    all_model_columns = X.columns.tolist()
    if target_col not in all_model_columns:
        print(f"Warning: Store type '{major_type}' not found in training data!")
        print(f"Available types: {[col.replace('Major Type_', '') for col in all_model_columns if col.startswith('Major Type_')]}")
        return None
        
    # Prepare input data with all features set to zero
    input_data = {col: 0 for col in X.columns}
    input_data['Sales SQ FT'] = sales_sqft
    # Set the dummy variable corresponding to the provided major_type to 1
    input_data[target_col] = 1
    
    # Perform prediction
    try:
        x_new = pd.DataFrame([input_data])
        prediction = reg.predict(x_new)[0]
        return round(prediction)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# === 8. Define a Function to Predict ESL Distribution by Region for a New Store ===
def predict_esl_distribution(sales_sqft, major_type=None):
    """
    Predict the ESL distribution by region for a new store.
    Uses the overall predicted total ESL and multiplies by the grouped average percentages for that Major Type.
    
    Parameters:
    -----------
    sales_sqft : float
        Sales area in square feet
    major_type : str, optional
        Store type. If None, will use the most common type in the dataset
        
    Returns:
    --------
    dict or None
        Dictionary mapping regions to predicted ESL counts
    """
    # Default to most common major type if none provided
    if major_type is None:
        major_type = df['Major Type'].value_counts().index[0]
        print(f"No store type specified. Using most common type: {major_type}")
    
    # First predict the overall total ESL for the new store
    total_prediction = predict_new_store(sales_sqft, major_type)
    if total_prediction is None:
        return None
        
    # Check if the major type is in the grouped averages
    try:
        all_types = grouped_avg.index.tolist()
        if major_type.upper() not in all_types:
            print(f"Warning: Store type '{major_type}' not found in grouped averages!")
            print(f"Available types: {all_types}")
            return None
            
        percentages = grouped_avg.loc[major_type.upper()]
        
        # Compute predicted ESL for each region
        region_predictions = {}
        for col in percentages.index:
            # Remove the " %" suffix to get the region name
            region_name = col.replace(" %", "")
            region_predictions[region_name] = round(total_prediction * percentages[col])
        return region_predictions
    except KeyError:
        print(f"Error: Major Type {major_type} not found in grouped averages. Check your data.")
        return None
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# === 9. Visualization Functions ===
def show_scatter_plot():
    """Generate and show scatter plot of Sales SQ FT vs Total ESL"""
    plt.figure(figsize=(12, 7))
    # Use the original dataframe (df) for plotting
    types = df['Major Type'].unique()
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']  # Different markers for different types
    for i, t in enumerate(types):
        subset = df[df['Major Type'] == t]
        plt.scatter(subset['Sales SQ FT'], subset['Total ESL'], 
                    label=t, alpha=0.8, s=80, 
                    marker=markers[i % len(markers)], 
                    edgecolor='white', linewidth=0.5)
        
    # Add regression line
    x_range = np.linspace(df['Sales SQ FT'].min(), df['Sales SQ FT'].max(), 100)
    plt.plot(x_range, reg.intercept_ + reg.coef_[0] * x_range, 
             color='black', linestyle='--', linewidth=2, label='Regression Line')

    plt.xlabel("Sales SQ FT", fontweight='bold')
    plt.ylabel("Total ESL", fontweight='bold')
    plt.title("Relationship Between Sales Area and Total ESL Count by Store Type", fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(title="Store Type", title_fontsize=12, framealpha=0.7)
    plt.tight_layout()
    plt.savefig("SalesSQFT_vs_TotalESL.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Scatter plot saved as SalesSQFT_vs_TotalESL.png")

def show_distribution_for_type(major_type=None):
    """Generate and show bar chart of ESL distribution for a specific store type"""
    # Default to most common major type if none provided
    if major_type is None:
        major_type = df['Major Type'].value_counts().index[0]
        print(f"No store type specified. Using most common type: {major_type}")
    
    try:
        all_types = grouped_avg.index.tolist()
        if major_type.upper() not in all_types:
            print(f"Warning: Store type '{major_type}' not found in grouped averages!")
            print(f"Available types: {all_types}")
            return False
            
        avg_percentages = grouped_avg.loc[major_type.upper()]
        # Prepare data for bar plot
        regions = [col.replace(" %", "") for col in avg_percentages.index]
        percentages = avg_percentages.values * 100  # Convert to percentage

        plt.figure(figsize=(14, 8))
        bars = plt.bar(regions, percentages, color=sns.color_palette("Set2", len(regions)))
        plt.xlabel("ESL Region", fontweight='bold')
        plt.ylabel("Average Percentage (%)", fontweight='bold')
        plt.title(f"Average ESL Percentage Distribution for {major_type} Stores", fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"Avg_ESL_Distribution_{major_type}.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Bar chart saved as Avg_ESL_Distribution_{major_type}.png")
        return True
    except KeyError:
        print(f"No grouped data for Major Type: {major_type}")
        return False
    except Exception as e:
        print(f"Error generating chart: {e}")
        return False

def show_all_types_comparison():
    """Generate and show comparison of ESL distributions across all store types"""
    # Get all available store types
    all_types = df['Major Type'].unique()
    
    # Create a figure with multiple subplots (one for each store type)
    n_types = len(all_types)
    cols = 2  # 2 columns of charts
    rows = (n_types + 1) // 2  # Calculate required rows (ceiling division)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    axes = axes.flatten()  # Flatten to easily iterate
    
    # Create a bar chart for each store type
    for i, store_type in enumerate(all_types):
        try:
            # Get percentage data for this store type
            avg_percentages = grouped_avg.loc[store_type]
            regions = [col.replace(" %", "") for col in avg_percentages.index]
            percentages = avg_percentages.values * 100  # Convert to percentage
            
            # Create bar chart in the appropriate subplot
            ax = axes[i]
            bars = ax.bar(regions, percentages, color=sns.color_palette("Set2", len(regions)))
            
            # Set labels and title
            ax.set_title(f"{store_type}", fontweight='bold')
            ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 5:  # Only show labels for bars with significant height
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                          f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=7)
        except KeyError:
            axes[i].text(0.5, 0.5, f"No data for {store_type}", 
                         ha='center', va='center', fontsize=14)
            
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    # Add common labels
    fig.text(0.5, 0.02, 'ESL Region', ha='center', fontweight='bold', fontsize=14)
    fig.text(0.02, 0.5, 'Average Percentage (%)', va='center', rotation='vertical', fontweight='bold', fontsize=14)
    fig.suptitle('ESL Distribution Comparison Across All Store Types', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig("All_Store_Types_ESL_Distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Comparison chart saved as All_Store_Types_ESL_Distribution.png")

# === 10. User Input and Predictions ===
def run_interactive_prediction():
    """Interactive tool to let users input parameters and display prediction results with visualization"""
    print("\n" + "="*50)
    print("ESL Prediction Tool - Interactive Version")
    print("="*50)
    
    # Display available store types
    available_types = list(df['Major Type'].unique())
    print("\nAvailable store types:")
    for i, type_name in enumerate(available_types, 1):
        print(f"{i}. {type_name}")
    
    # Get user input
    while True:
        try:
            sales_sqft = float(input("\nEnter sales area (Sales SQ FT): "))
            if sales_sqft <= 0:
                print("Sales area must be a positive number! Please try again.")
                continue
                
            print("\nSelect store type (enter number or type name):")
            type_input = input("> ")
            
            # Process input - can be either number or type name
            selected_type = None
            if type_input.isdigit():
                idx = int(type_input) - 1
                if 0 <= idx < len(available_types):
                    selected_type = available_types[idx]
                else:
                    print(f"Invalid number! Please enter a number between 1 and {len(available_types)}.")
                    continue
            else:
                # Direct type name input - check for match (case insensitive)
                for type_name in available_types:
                    if type_input.upper() == type_name.upper():
                        selected_type = type_name
                        break
                
                if not selected_type:
                    print("Store type not found! Please try again.")
                    continue
            
            break
        except ValueError:
            print("Invalid input! Please enter a number.")
    
    # Display selected parameters
    print("\n" + "-"*50)
    print(f"Selected parameters: Sales area = {sales_sqft:.2f} SQ FT | Store type = {selected_type}")
    print("-"*50)
    
    # Generate prediction results
    total_prediction = predict_new_store(sales_sqft, major_type=selected_type)
    print(f"\n🔮 Predicted total ESL count: {total_prediction}")
    
    # Predict ESL distribution by region
    region_prediction = predict_esl_distribution(sales_sqft, major_type=selected_type)
    if region_prediction:
        print("\n🔮 Predicted ESL distribution by region:")
        # Create table-style output for better readability
        max_region_length = max(len(region) for region in region_prediction.keys())
        print(f"{'Region':<{max_region_length+5}}{'Count':<10}{'Percentage':<10}")
        print("-" * (max_region_length + 25))
        
        for region, count in region_prediction.items():
            percentage = (count / total_prediction) * 100
            print(f"{region:<{max_region_length+5}}{count:<10}{percentage:.2f}%")
        
        # Generate visualization
        regions = list(region_prediction.keys())
        counts = list(region_prediction.values())
        plt.figure(figsize=(14, 8))
        bars = plt.bar(regions, counts, color=sns.color_palette("viridis", len(regions)))
        plt.xlabel("ESL Region", fontweight='bold')
        plt.ylabel("Predicted ESL Count", fontweight='bold')
        plt.title(f"Predicted ESL Distribution - {selected_type} - {sales_sqft:.0f} SQ FT", fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        result_filename = f"ESL_Prediction_{selected_type}_{int(sales_sqft)}_SQFT.png"
        plt.savefig(result_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\nPrediction chart saved as {result_filename}")
        
        # Ask if user wants to see additional visualizations
        print("\nWould you like to see additional visualizations?")
        print("1. Show scatter plot (Sales SQ FT vs Total ESL)")
        print("2. Show average distribution for selected store type")
        print("3. Show comparison of all store types")
        print("4. Exit")
        try:
            viz_choice = int(input("> "))
            if viz_choice == 1:
                show_scatter_plot()
            elif viz_choice == 2:
                show_distribution_for_type(selected_type)
            elif viz_choice == 3:
                show_all_types_comparison()
        except (ValueError, IndexError):
            pass

# Execute interactive prediction
if __name__ == "__main__":
    # 直接运行交互式预测，不再显示默认示例
    run_interactive_prediction()
