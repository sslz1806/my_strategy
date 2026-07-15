@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8

echo 开始执行脚本...
cd /d C:\Users\20561\Desktop\策略

REM ====== 1. 基础数据更新（数据更新v2） ======
echo [%date% %time%] 正在更新基础数据...
E:\working\anaconda3\envs\quant\python.exe 任务\数据更新v2.py
if errorlevel 1 (
    echo 数据更新v2.py 执行出错（退出码=%errorlevel%），请检查日志！
    goto :end
)
echo %date% %time% - 数据更新v2 完成 >> log\update_data.log

REM ====== 2. 米筐数据更新 ======
echo [%date% %time%] 正在更新米筐数据...
E:\working\anaconda3\envs\quant\python.exe 任务\米筐数据更新.py
if errorlevel 1 (
    echo 米筐数据更新.py 执行出错（退出码=%errorlevel%），请检查日志！
    goto :end
)
echo %date% %time% - 米筐数据更新 完成 >> log\update_data.log

echo 所有脚本已成功执行完毕！

:end
pause