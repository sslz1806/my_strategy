chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8

@echo off
echo 开始执行脚本...
cd /d C:\Users\20561\Desktop\策略
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py
echo %date% %time% - 数据更新完成 >> log\update_data.log
type con >> log\update_data.log 2>&1

if %errorlevel% neq 0 (
    echo 脚本执行过程中发生错误！
    goto :end
)

echo 脚本已成功执行完毕！

:end
pause