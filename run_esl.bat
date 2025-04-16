@echo off
echo === 启动ESL预测项目 ===
echo 正在激活虚拟环境...

REM 使用CMD激活虚拟环境(不使用PowerShell)
call esl_env\Scripts\activate.bat

REM 启动预测程序
python esl_prediction_advanced.py

pause 