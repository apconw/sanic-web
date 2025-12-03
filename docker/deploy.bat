@echo off
chcp 65001 >nul
echo ========================================
echo   Sanic-Web Docker 部署脚本
echo ========================================
echo.
echo 正在启动 PowerShell 脚本...
echo.

powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0deploy_and_init.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo   部署过程中出现错误！
    echo   错误代码: %ERRORLEVEL%
    echo ========================================
    echo.
    echo 提示：如果遇到权限问题，请右键选择"以管理员身份运行"
    echo.
)

pause