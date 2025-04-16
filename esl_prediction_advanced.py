import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置更好看的图表样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# === 1. 读取和清洗数据 ===
print("读取数据中...")
file_path = "DSL Data in Cincinnati.xlsx"
df = pd.read_excel(file_path, sheet_name='工作表1', header=1)
df.columns = df.columns.astype(str).str.strip()

# === 2. 定义ESL区域列 ===
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

# 将指定列转换为数值类型
for col in esl_columns + ['Total', 'Sales SQ FT']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 创建Total ESL列
df['Total ESL'] = df['Total']

# === 3. 计算每个ESL区域的百分比 ===
for col in esl_columns:
    df[f"{col} %"] = df[col] / df['Total ESL']

# === 4. 按店铺类型分组统计 ===
grouped_avg = df.groupby('Major Type')[[f"{col} %" for col in esl_columns]].mean()
grouped_avg.to_excel("ESL_Grouped_Averages_by_Type.xlsx")
print("按商店类型分组的平均值已保存到 ESL_Grouped_Averages_by_Type.xlsx")

# === 5. 特征工程 ===
print("\n执行特征工程...")
df_model = df.copy()
# 过滤缺失值
df_model = df_model[df_model['Sales SQ FT'].notnull() & df_model['Total ESL'].notnull()]

# 添加新特征
df_model['ESL_per_SqFt'] = df_model['Total ESL'] / df_model['Sales SQ FT']
df_model['Log_Sales_SqFt'] = np.log1p(df_model['Sales SQ FT'])
df_model['Sqrt_Sales_SqFt'] = np.sqrt(df_model['Sales SQ FT'])

# 区域计数特征
df_model['ESL_Regions_Count'] = df_model[esl_columns].count(axis=1)
df_model['ESL_1.54_Total'] = df_model[[col for col in esl_columns if '1.54"' in col]].sum(axis=1)
df_model['ESL_2.66_Total'] = df_model[[col for col in esl_columns if '2.66"' in col]].sum(axis=1)
df_model['ESL_4.20_Total'] = df_model[[col for col in esl_columns if '4.20"' in col]].sum(axis=1)

# 创建虚拟变量
df_model = pd.get_dummies(df_model, columns=['Major Type'])

# 探索性数据分析
print("数据形状:", df_model.shape)
print("\n主要统计量:")
print(df_model[['Sales SQ FT', 'Total ESL', 'ESL_per_SqFt']].describe())

# 检测异常值 (可选择后续处理)
Q1 = df_model['Total ESL'].quantile(0.25)
Q3 = df_model['Total ESL'].quantile(0.75)
IQR = Q3 - Q1
outliers = df_model[(df_model['Total ESL'] < (Q1 - 1.5 * IQR)) | (df_model['Total ESL'] > (Q3 + 1.5 * IQR))]
print(f"\n检测到 {len(outliers)} 个潜在异常值")

# === 6. 准备模型数据 ===
print("\n准备模型数据...")
# 选择特征
feature_cols = ['Sales SQ FT', 'Log_Sales_SqFt', 'Sqrt_Sales_SqFt', 
                'ESL_Regions_Count', 'ESL_1.54_Total', 'ESL_2.66_Total', 'ESL_4.20_Total'] + \
               [col for col in df_model.columns if col.startswith('Major Type_')]

X = df_model[feature_cols]
y = df_model['Total ESL']

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练数据大小: {X_train.shape[0]} 个样本")
print(f"测试数据大小: {X_test.shape[0]} 个样本")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 7. 训练多个模型并比较性能 ===
print("\n训练和评估模型中...")

# 创建用于比较的模型字典
models = {
    "Linear Regression": LinearRegression(),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}

# 存储结果
model_results = []
trained_models = {}

# 训练和评估每个模型
for name, model in models.items():
    print(f"\n正在训练 {name} 模型...")
    
    # 训练模型 - 对于部分模型使用缩放数据
    if name in ["ElasticNet", "SVR"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # 计算性能指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 执行交叉验证
    if name in ["ElasticNet", "SVR"]:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # 存储结果
    result = {
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "CV R² (mean)": cv_scores.mean(),
        "CV R² (std)": cv_scores.std()
    }
    model_results.append(result)
    trained_models[name] = model
    
    # 输出结果
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  5折交叉验证 R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 创建性能比较表
results_df = pd.DataFrame(model_results)
print("\n模型性能比较:")
print(results_df[['Model', 'RMSE', 'R²', 'CV R² (mean)']].sort_values('R²', ascending=False))

# 选择最佳模型
best_model_name = results_df.sort_values('R²', ascending=False).iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\n✓ 最佳模型: {best_model_name} (R² = {results_df[results_df['Model'] == best_model_name]['R²'].values[0]:.4f})")

# 保存最佳模型
models_dir = "advanced_models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"{models_dir}/esl_prediction_{best_model_name.replace(' ', '_').lower()}_{timestamp}.joblib"

# 保存模型和缩放器
model_data = {
    'model': best_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'model_name': best_model_name
}
joblib.dump(model_data, model_filename)
print(f"\n💾 最佳模型已保存到 {model_filename}")

# === 8. 可视化模型比较 ===
def visualize_model_comparison():
    """比较不同模型的性能"""
    # 性能指标比较
    plt.figure(figsize=(12, 6))
    
    # R² 值比较
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='R²', data=results_df, palette='viridis')
    plt.title('模型 R² 比较')
    plt.ylim(max(0, results_df['R²'].min() - 0.1), 1.0)
    plt.xticks(rotation=45, ha='right')
    
    # RMSE 比较
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='RMSE', data=results_df, palette='viridis')
    plt.title('模型 RMSE 比较')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("Model_Comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("模型比较图表已保存为 Model_Comparison.png")

def visualize_predictions():
    """比较预测值与实际值"""
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    if best_model_name in ["ElasticNet", "SVR"]:
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)
    
    # 散点图
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    
    # 理想线
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'{best_model_name} 模型: 预测值 vs 实际值')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Prediction_vs_Actual.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("预测值与实际值比较图已保存为 Prediction_vs_Actual.png")

def visualize_feature_importance():
    """显示特征重要性 (适用于树模型)"""
    if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        plt.figure(figsize=(10, 8))
        
        # 获取特征重要性
        if best_model_name == "XGBoost":
            importance = best_model.feature_importances_
        else:
            importance = best_model.feature_importances_
            
        # 创建特征重要性DataFrame
        feat_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # 绘制条形图
        sns.barplot(x='Importance', y='Feature', data=feat_importance, palette='viridis')
        plt.title(f'{best_model_name} - 特征重要性')
        plt.tight_layout()
        plt.savefig("Feature_Importance_Advanced.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("特征重要性图表已保存为 Feature_Importance_Advanced.png")
        return feat_importance
    else:
        print(f"注意: {best_model_name} 模型不提供直接的特征重要性。")
        return None

def explore_residuals():
    """分析残差"""
    if best_model_name in ["ElasticNet", "SVR"]:
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)
    
    residuals = y_test - y_pred
    
    plt.figure(figsize=(16, 6))
    
    # 残差分布
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title('残差分布')
    plt.xlabel('残差')
    
    # 残差与预测值关系
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('残差 vs 预测值')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    
    plt.tight_layout()
    plt.savefig("Residual_Analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("残差分析图表已保存为 Residual_Analysis.png")

# === 9. 改进的预测函数 ===
def advanced_predict(sales_sqft, major_type=None, model_data=model_data):
    """
    基于高级模型预测ESL总数
    
    参数:
    ------
    sales_sqft : float
        销售面积 (平方英尺)
    major_type : str, optional
        商店类型。如果为None，将使用数据集中最常见的类型  
    model_data : dict
        包含模型和相关信息的字典
        
    返回:
    ------
    int
        预测的ESL总数
    """
    # 默认为最常见的商店类型
    if major_type is None:
        major_type = df['Major Type'].value_counts().index[0]
        print(f"未指定商店类型。使用最常见类型: {major_type}")
    
    # 提取模型和相关数据
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    model_name = model_data['model_name']
    
    # 准备输入数据
    input_data = {col: 0 for col in feature_cols}
    
    # 设置特征值
    input_data['Sales SQ FT'] = sales_sqft
    input_data['Log_Sales_SqFt'] = np.log1p(sales_sqft)
    input_data['Sqrt_Sales_SqFt'] = np.sqrt(sales_sqft)
    
    # 设置虚拟变量
    target_col = f'Major Type_{major_type.upper()}'
    if target_col in input_data:
        input_data[target_col] = 1
    else:
        available_types = [col.replace('Major Type_', '') for col in feature_cols if col.startswith('Major Type_')]
        print(f"警告: 商店类型 '{major_type}' 在训练数据中未找到!")
        print(f"可用类型: {available_types}")
        return None
    
    # 设置区域计数默认值 (使用类似商店的平均值)
    similar_stores = df[df['Major Type'] == major_type]
    if len(similar_stores) > 0:
        input_data['ESL_Regions_Count'] = similar_stores['Sales SQ FT'].count(axis=1).mean()
        # 估算各种尺寸的ESL总数 (基于相似商店的平均值)
        small_esl_cols = [col for col in esl_columns if '1.54"' in col]
        medium_esl_cols = [col for col in esl_columns if '2.66"' in col]
        large_esl_cols = [col for col in esl_columns if '4.20"' in col]
        
        if len(small_esl_cols) > 0:
            input_data['ESL_1.54_Total'] = similar_stores[small_esl_cols].sum(axis=1).mean()
        if len(medium_esl_cols) > 0:
            input_data['ESL_2.66_Total'] = similar_stores[medium_esl_cols].sum(axis=1).mean()
        if len(large_esl_cols) > 0:
            input_data['ESL_4.20_Total'] = similar_stores[large_esl_cols].sum(axis=1).mean()
    else:
        # 如果没有相似商店，使用整体平均值
        input_data['ESL_Regions_Count'] = len(esl_columns)
        input_data['ESL_1.54_Total'] = 0
        input_data['ESL_2.66_Total'] = 0
        input_data['ESL_4.20_Total'] = 0
    
    # 创建输入DataFrame
    x_new = pd.DataFrame([input_data])
    
    # 根据模型类型预处理数据
    if model_name in ["ElasticNet", "SVR"]:
        x_new_processed = scaler.transform(x_new)
    else:
        x_new_processed = x_new
    
    # 执行预测
    try:
        prediction = model.predict(x_new_processed)[0]
        return max(0, round(prediction))  # 确保预测值为非负整数
    except Exception as e:
        print(f"预测错误: {e}")
        return None

def advanced_predict_distribution(sales_sqft, major_type=None):
    """
    预测新商店的ESL区域分布
    
    参数与advanced_predict相同
    """
    # 首先预测总ESL数量
    total_prediction = advanced_predict(sales_sqft, major_type)
    if total_prediction is None:
        return None
    
    # 默认为最常见的商店类型
    if major_type is None:
        major_type = df['Major Type'].value_counts().index[0]
    
    # 检查商店类型是否在分组平均值中存在
    try:
        all_types = grouped_avg.index.tolist()
        if major_type.upper() not in all_types:
            print(f"警告: 商店类型 '{major_type}' 在分组平均值中未找到!")
            print(f"可用类型: {all_types}")
            return None
        
        # 获取该类型的百分比分布
        percentages = grouped_avg.loc[major_type.upper()]
        
        # 计算各区域预测值
        region_predictions = {}
        for col in percentages.index:
            region_name = col.replace(" %", "")
            region_predictions[region_name] = round(total_prediction * percentages[col])
        
        return region_predictions
    except KeyError:
        print(f"错误: 商店类型 {major_type} 在分组平均值中未找到。请检查数据。")
        return None
    except Exception as e:
        print(f"预测错误: {e}")
        return None

# === 10. 交互式预测工具 ===
def run_advanced_prediction():
    """交互式工具，允许用户输入参数并显示预测结果与可视化"""
    # 主循环 - 直到用户选择退出
    while True:
        print("\n" + "="*50)
        print("ESL 高级预测工具")
        print("="*50)
        
        # 显示可用的商店类型
        available_types = list(df['Major Type'].unique())
        print("\n可用的商店类型:")
        for i, type_name in enumerate(available_types, 1):
            print(f"{i}. {type_name}")
        
        # 获取用户输入
        while True:
            try:
                sales_sqft = float(input("\n输入销售面积 (Sales SQ FT): "))
                if sales_sqft <= 0:
                    print("销售面积必须为正数! 请重试。")
                    continue
                
                print("\n选择商店类型 (输入编号或类型名称):")
                type_input = input("> ")
                
                # 处理输入 - 可以是编号或类型名称
                selected_type = None
                if type_input.isdigit():
                    idx = int(type_input) - 1
                    if 0 <= idx < len(available_types):
                        selected_type = available_types[idx]
                    else:
                        print(f"无效编号! 请输入1到{len(available_types)}之间的数字。")
                        continue
                else:
                    # 直接输入类型名称 - 检查匹配 (不区分大小写)
                    for type_name in available_types:
                        if type_input.upper() == type_name.upper():
                            selected_type = type_name
                            break
                    
                    if not selected_type:
                        print("未找到该商店类型! 请重试。")
                        continue
                
                break
            except ValueError:
                print("无效输入! 请输入数字。")
        
        # 显示选择的参数
        print("\n" + "-"*50)
        print(f"选择的参数: 销售面积 = {sales_sqft:.2f} SQ FT | 商店类型 = {selected_type}")
        print("-"*50)
        
        # 使用高级模型生成预测结果
        total_prediction = advanced_predict(sales_sqft, major_type=selected_type)
        print(f"\n🔮 预测总ESL数量: {total_prediction}")
        
        # 预测ESL区域分布
        region_prediction = advanced_predict_distribution(sales_sqft, major_type=selected_type)
        if region_prediction:
            print("\n🔮 预测的ESL区域分布:")
            # 创建表格风格输出，提高可读性
            max_region_length = max(len(region) for region in region_prediction.keys())
            print(f"{'区域':<{max_region_length+5}}{'数量':<10}{'百分比':<10}")
            print("-" * (max_region_length + 25))
            
            for region, count in region_prediction.items():
                percentage = (count / total_prediction) * 100
                print(f"{region:<{max_region_length+5}}{count:<10}{percentage:.2f}%")
            
            # 生成可视化
            regions = list(region_prediction.keys())
            counts = list(region_prediction.values())
            plt.figure(figsize=(14, 8))
            bars = plt.bar(regions, counts, color=sns.color_palette("viridis", len(regions)))
            plt.xlabel("ESL 区域", fontweight='bold')
            plt.ylabel("预测的 ESL 数量", fontweight='bold')
            plt.title(f"预测的 ESL 分布 - {selected_type} - {sales_sqft:.0f} SQ FT", fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # 在每个柱上添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            result_filename = f"Advanced_ESL_Prediction_{selected_type}_{int(sales_sqft)}_SQFT.png"
            plt.savefig(result_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"\n预测图表已保存为 {result_filename}")
        
        # 菜单选项
        while True:
            print("\n您想做什么?")
            print("1. 进行另一次预测")
            print("2. 查看当前预测的其他分析")
            print("3. 查看模型性能比较")
            print("4. 退出程序")
            
            try:
                choice = int(input("> "))
                if choice == 1:
                    # 退出内部菜单循环，继续外部预测循环进行新预测
                    break
                elif choice == 2:
                    # 显示可视化选项子菜单
                    print("\n选择分析图表:")
                    print("1. 显示预测值与实际值比较")
                    print("2. 显示特征重要性分析")
                    print("3. 显示残差分析")
                    print("4. 返回主菜单")
                    
                    try:
                        viz_choice = int(input("> "))
                        if viz_choice == 1:
                            visualize_predictions()
                        elif viz_choice == 2:
                            feature_imp = visualize_feature_importance()
                            if feature_imp is not None:
                                print("\n前5个最重要特征:")
                                print(feature_imp.head(5))
                        elif viz_choice == 3:
                            explore_residuals()
                        # 选项4只是继续循环
                    except (ValueError, IndexError):
                        print("无效选择。请重试。")
                elif choice == 3:
                    visualize_model_comparison()
                elif choice == 4:
                    print("\n退出ESL预测工具。再见!")
                    return  # 完全退出函数
                else:
                    print("无效选择。请重试。")
            except (ValueError, IndexError):
                print("无效输入。请输入数字。")

# === 11. 加载保存的模型 ===
def load_model(model_path=None):
    """加载之前保存的模型或使用当前模型"""
    global model_data
    if model_path and os.path.exists(model_path):
        try:
            loaded_model_data = joblib.load(model_path)
            model_data = loaded_model_data
            print(f"已从 {model_path} 加载模型")
            return True
        except Exception as e:
            print(f"加载模型错误: {e}")
            return False
    return False

# === 12. 可视化模型比较和性能分析 ===
print("\n生成模型比较可视化...")
visualize_model_comparison()
visualize_predictions()
feature_importance = visualize_feature_importance()
if feature_importance is not None:
    print("\n最重要的5个特征:")
    print(feature_importance.head(5))
explore_residuals()

# === 13. 执行交互式预测 ===
if __name__ == "__main__":
    # 检查是否应加载保存的模型
    if os.path.exists("advanced_models"):
        model_files = [f for f in os.listdir("advanced_models") if f.endswith(".joblib")]
        if model_files:
            print("\n找到已保存的模型:")
            for i, model_file in enumerate(model_files, 1):
                print(f"{i}. {model_file}")
            print(f"{len(model_files)+1}. 使用当前模型")
            
            try:
                choice = int(input("\n选择要使用的模型: "))
                if 1 <= choice <= len(model_files):
                    load_model(os.path.join("advanced_models", model_files[choice-1]))
            except (ValueError, IndexError):
                print("无效选择。使用当前模型。")
    
    # 运行交互式预测
    print("\n启动交互式预测工具...")
    run_advanced_prediction() 