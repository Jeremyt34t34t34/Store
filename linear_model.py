import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# # 步骤 1：读取数据，设置正确的标题行
# file_path = "DSL Data in Cincinnati.xlsx"  # 请确认此文件与脚本在同一目录
# df = pd.read_excel(file_path, sheet_name='工作表1', header=1)

# # 步骤 2：清洗列名
# df.columns = df.columns.astype(str).str.strip()

# # 步骤 3：定义需要计算比例的 ESL 字段
# esl_columns = [
#     '1.54"', '2.66"', '4.20"',
#     'Central Market 1.54"', 'Central Market 2.66"',
#     'Checkout 1.54"', 'Checkout 2.66"',
#     'Pharmacy 1.54"', 'Pharmacy 2.66"',
#     'Beer 1.54"', 'Beer 2.66"',
#     'Cosmetics 1.54"',
#     'Dairy 1.54"', 'Dairy 2.66"',
#     'Seafood & Meat 1.54"', 'Seafood & Meat 2.66"', 'Seafood & Meat 4.20"',
#     'Produce 1.54"', 'Produce 2.66"', 'Produce 4.20"',
#     'Bakery 1.54"', 'Bakery 2.66"'
# ]

# # 步骤 4：确保字段可数值化
# for col in esl_columns + ['Total']:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # 步骤 5：添加一列“Total ESL”用于分母
# df['Total ESL'] = df['Total']

# # 步骤 6：计算占比
# for col in esl_columns:
#     df[f"{col} %"] = df[col] / df['Total ESL']

# # 步骤 7：保存结果到 Excel
# output_path = "ESL_Proportion_Output.xlsx"
# df.to_excel(output_path, index=False)

# print(f"✅ ESL 占比计算完成，输出文件为：{output_path}")

# === 1. Read and clean data ===
file_path = "DSL Data in Cincinnati.xlsx"
# Read the Excel file; header row is the second row (index=1)
df = pd.read_excel(file_path, sheet_name='工作表1', header=1)
# Clean column names by stripping extra whitespace
df.columns = df.columns.astype(str).str.strip()

# === 2. Define ESL region columns ===
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

# === 3. Calculate percentage proportion for each ESL region ===
for col in esl_columns:
    df[f"{col} %"] = df[col] / df['Total ESL']

# === 4. Group statistics: average ESL distribution by Major Type ===
grouped_avg = df.groupby('Major Type')[[f"{col} %" for col in esl_columns]].mean()
# Save the grouped average to an Excel file
grouped_avg.to_excel("ESL_Grouped_Averages_by_Type.xlsx")
# === 5. Build a linear regression model to predict Total ESL using Sales SQ FT and Major Type ===
# Filter out records with missing values
df_model = df.copy()
df_model = df_model[df_model['Sales SQ FT'].notnull() & df_model['Total ESL'].notnull()]
# Create dummy variables for 'Major Type'
df_model = pd.get_dummies(df_model, columns=['Major Type'])

# Define features: Sales SQ FT and dummy columns for Major Type
X = df_model[['Sales SQ FT'] + [col for col in df_model.columns if col.startswith('Major Type_')]]
y = df_model['Total ESL']

# Train the linear regression model
reg = LinearRegression()
reg.fit(X, y)

# === 6. Output model coefficients ===
print("✅ Linear Regression Model Coefficients:")
for name, coef in zip(X.columns, reg.coef_):
    print(f"{name}: {coef:.2f}")
print(f"Intercept: {reg.intercept_:.2f}")

# === 7. Define a prediction function for a new store ===
def predict_new_store(sales_sqft, major_type='COMBINATION'):
    """
    Predict total ESL count for a new store given its Sales SQ FT and Major Type.
    """
    # Prepare input data with all features set to zero
    input_data = {col: 0 for col in X.columns}
    input_data['Sales SQ FT'] = sales_sqft
    # Set the dummy variable corresponding to the provided major_type to 1
    target_col = f'Major Type_{major_type.upper()}'
    if target_col in input_data:
        input_data[target_col] = 1
    # Create a DataFrame for the input and predict
    x_new = pd.DataFrame([input_data])
    prediction = reg.predict(x_new)[0]
    return round(prediction)

# === 8. Example prediction for a new store ===
example_prediction = predict_new_store(55000, major_type='COMBINATION')
print(f"🔮 Predicted ESL total for a COMBINATION store with 55,000 SQ FT of Sales area: {example_prediction}")