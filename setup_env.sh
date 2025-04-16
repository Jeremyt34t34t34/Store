#!/bin/bash
echo "=== 创建ESL预测项目虚拟环境 ==="

# 创建虚拟环境
python3 -m venv esl_env
echo "虚拟环境已创建!"

# 激活虚拟环境
source esl_env/bin/activate
echo "虚拟环境已激活!"

# 安装所需的库
echo "正在安装库..."
pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost openpyxl

echo "=== 安装完成! ==="
echo "使用以下命令激活环境:"
echo "source esl_env/bin/activate"
echo ""
echo "使用以下命令运行程序:"
echo "python esl_prediction_advanced.py"
echo "" 