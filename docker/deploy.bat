@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ========================================
echo   Sanic-Web Docker 部署脚本
echo ========================================
echo.
echo 正在启动 PowerShell 脚本...
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 检查 PowerShell 版本并执行脚本
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& {Set-Location -LiteralPath '%CD%'; & '%~dp0deploy_and_init.ps1'}"

set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% NEQ 0 (
    echo.
    echo ========================================
    echo   部署过程中出现错误！
    echo   错误代码: %EXIT_CODE%
    echo ========================================
    echo.
    echo 提示：如果遇到权限问题，请右键选择"以管理员身份运行"
    echo.
)

pause
exit /b %EXIT_CODE%