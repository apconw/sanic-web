@echo off
setlocal enabledelayedexpansion

echo 🚀 开始部署和初始化流程...

:: 日志记录函数
goto :main

:log_error
echo ❌ 错误: %~1
echo %date% %time%: 错误 - %~1 >> error.log
exit /b

:log_info
echo %~1
echo %date% %time%: 信息 - %~1 >> deploy.log
exit /b

:main
:: 1. 创建volume目录和mcp_settings.json文件
echo 📁 创建volume目录和配置文件...
if not exist ".\volume\mcp-data" (
    mkdir ".\volume\mcp-data" 2>nul
    if errorlevel 1 (
        call :log_error "无法创建目录 .\volume\mcp-data"
    )
)

if not exist ".\volume\mcp-data\mcp_settings.json" (
    type nul > ".\volume\mcp-data\mcp_settings.json" 2>nul
    if errorlevel 1 (
        call :log_error "无法创建文件 .\volume\mcp-data\mcp_settings.json"
    )
)

:: 2. 启动所有服务
call :log_info "🐳 启动Docker服务..."
docker-compose up -d
if errorlevel 1 (
    call :log_error "Docker服务启动失败"
)

:: 3. 检查Python环境
call :log_info "🔍 检查Python环境..."
where python >nul 2>nul
if errorlevel 1 (
    call :log_error "未检测到Python环境"
    call :log_info "请先安装Python:"
    call :log_info "访问 https://www.python.org/downloads/ 下载安装包"
    call :log_info "安装时请勾选 'Add Python to PATH' 选项"
)

pip --version >nul 2>nul
if errorlevel 1 (
    call :log_error "未检测到pip工具"
    call :log_info "请先安装Python，pip应该随Python一起安装"
)

python --version >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    call :log_info "✅ Python环境检查通过 (版本: !PYTHON_VERSION!)"
)

:: 4. 安装Python依赖
call :log_info "🐍 安装Python依赖..."
pip install pymysql py2neo
if errorlevel 1 (
    call :log_error "Python依赖安装失败"
)

:: 5. 检查容器是否运行
call :log_info "🔍 检查服务启动状态..."

:: 等待容器启动函数
goto :check_container
:wait_for_container
set CONTAINER_NAME=%~1
set MAX_ATTEMPTS=30
set ATTEMPT=1

call :log_info "⏳ 等待 %CONTAINER_NAME% 启动..."

:wait_loop
if !ATTEMPT! GTR !MAX_ATTEMPTS! (
    call :log_error "%CONTAINER_NAME% 启动超时"
    exit /b 1
)

docker inspect -f {{.State.Running}} %CONTAINER_NAME% 2>nul | findstr "true" >nul
if not errorlevel 1 (
    call :log_info "✅ %CONTAINER_NAME% 已成功启动"
    exit /b 0
)

call :log_info "⏳ %CONTAINER_NAME% 尚未启动，第 !ATTEMPT!/!MAX_ATTEMPTS! 次尝试..."
set /a ATTEMPT+=1
timeout /t 10 /nobreak >nul
goto :wait_loop

:: 检查MySQL服务是否真正可用
goto :check_mysql_ready
:check_mysql_ready
set MAX_ATTEMPTS=30
set ATTEMPT=1

call :log_info "⏳ 等待 MySQL 服务准备就绪..."

:mysql_ready_loop
if !ATTEMPT! GTR !MAX_ATTEMPTS! (
    call :log_error "MySQL 服务准备超时"
    exit /b 1
)

docker exec chat-db mysqladmin ping --silent >nul 2>&1
if not errorlevel 1 (
    call :log_info "✅ MySQL 服务已准备就绪"
    exit /b 0
)

call :log_info "⏳ MySQL 尚未准备就绪，第 !ATTEMPT!/!MAX_ATTEMPTS! 次尝试..."
set /a ATTEMPT+=1
timeout /t 5 /nobreak >nul
goto :mysql_ready_loop

:: 检查指定端口是否可用
goto :check_port
:check_port_available
set SERVICE_NAME=%~1
set PORT=%~2
set MAX_ATTEMPTS=30
set ATTEMPT=1

call :log_info "⏳ 检查 %SERVICE_NAME% 端口 %PORT% 是否可用..."

:port_check_loop
if !ATTEMPT! GTR !MAX_ATTEMPTS! (
    call :log_error "%SERVICE_NAME% 端口 %PORT% 检查超时"
    exit /b 1
)

netstat -an | findstr ":%PORT% " | findstr "LISTENING" >nul
if not errorlevel 1 (
    call :log_info "✅ %SERVICE_NAME% 端口 %PORT% 已开放"
    exit /b 0
)

call :log_info "⏳ %SERVICE_NAME% 端口 %PORT% 尚未开放，第 !ATTEMPT!/!MAX_ATTEMPTS! 次尝试..."
set /a ATTEMPT+=1
timeout /t 5 /nobreak >nul
goto :port_check_loop

:: 返回主流程
goto :main_continued
:check_container
call :wait_for_container "chat-db"
set CONTAINER_MYSQL_OK=!errorlevel!

call :wait_for_container "neo4j-apoc"
set CONTAINER_NEO4J_OK=!errorlevel!

call :check_mysql_ready
set MYSQL_READY_OK=!errorlevel!

call :check_port_available "MySQL" 13006
set PORT_MYSQL_OK=!errorlevel!

call :check_port_available "Neo4j" 7687
set PORT_NEO4J_OK=!errorlevel!

:main_continued
:: 6. 执行数据初始化脚本
if "!CONTAINER_MYSQL_OK!"=="0" if "!CONTAINER_NEO4J_OK!"=="0" if "!MYSQL_READY_OK!"=="0" if "!PORT_MYSQL_OK!"=="0" if "!PORT_NEO4J_OK!"=="0" (
    call :log_info "📊 等待服务稳定 (30秒)..."
    timeout /t 30 /nobreak >nul

    call :log_info "📊 执行数据初始化..."
    if exist ".\init_data.bat" (
        :: 添加重试机制
        set MAX_ATTEMPTS=3
        set ATTEMPT=1

        :retry_loop
        call .\init_data.bat
        if not errorlevel 1 (
            call :log_info "🎉 部署和初始化完成！"
            goto :end
        ) else (
            if !ATTEMPT! GEQ !MAX_ATTEMPTS! (
                call :log_error "初始化失败，已重试 !MAX_ATTEMPTS! 次"
                exit /b 1
            ) else (
                call :log_info "⚠️  初始化失败，第 !ATTEMPT! 次重试..."
                set /a ATTEMPT+=1
                timeout /t 10 /nobreak >nul
                goto :retry_loop
            )
        )
    ) else (
        call :log_error "初始化脚本 init_data.bat 不存在"
    )
) else (
    call :log_error "服务启动失败，无法执行数据初始化"
    call :log_info "各服务状态:"
    call :log_info "- MySQL容器启动: !CONTAINER_MYSQL_OK!"
    call :log_info "- Neo4j容器启动: !CONTAINER_NEO4J_OK!"
    call :log_info "- MySQL服务就绪: !MYSQL_READY_OK!"
    call :log_info "- MySQL端口可用: !PORT_MYSQL_OK!"
    call :log_info "- Neo4j端口可用: !PORT_NEO4J_OK!"
)

:end
echo.
echo 按任意键退出...
pause >nul
