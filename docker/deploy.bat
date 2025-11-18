@echo off
setlocal enabledelayedexpansion

call :log_info "🚀 开始部署和初始化流程..."

:: =============== 函数定义 ===============
:log_error
echo ❌ 错误: %~1
echo %date% %time%: 错误 - %~1 >> error.log
exit /b 1

:log_info
echo [%date% %time%] %~1
echo %date% %time%: 信息 - %~1 >> deploy.log
exit /b

:wait_for_container
set "CONTAINER_NAME=%~1"
set "MAX_ATTEMPTS=30"
set "ATTEMPT=1"

call :log_info "⏳ 等待容器 '!CONTAINER_NAME!' 启动..."

:wait_loop
if !ATTEMPT! GTR !MAX_ATTEMPTS! (
    call :log_error "容器 '!CONTAINER_NAME!' 启动超时"
)

for /f "tokens=*" %%s in ('docker inspect -f "{{.State.Running}}" "!CONTAINER_NAME!" 2^>nul') do set "STATUS=%%s"
if /i "!STATUS!"=="true" (
    call :log_info "✅ 容器 '!CONTAINER_NAME!' 已运行"
    exit /b 0
)

call :log_info "⏳ 容器 '!CONTAINER_NAME!' 尚未运行（第 !ATTEMPT!/!MAX_ATTEMPTS! 次尝试）..."
set /a ATTEMPT+=1
timeout /t 5 /nobreak >nul
goto :wait_loop

:check_mysql_ready
set "MAX_ATTEMPTS=30"
set "ATTEMPT=1"
call :log_info "⏳ 等待 MySQL 服务就绪..."

:mysql_ready_loop
if !ATTEMPT! GTR !MAX_ATTEMPTS! (
    call :log_error "MySQL 服务准备超时"
)

docker exec chat-db mysqladmin ping --silent >nul 2>&1
if not errorlevel 1 (
    call :log_info "✅ MySQL 服务已就绪"
    exit /b 0
)

call :log_info "⏳ MySQL 尚未就绪（第 !ATTEMPT!/!MAX_ATTEMPTS! 次尝试）..."
set /a ATTEMPT+=1
timeout /t 5 /nobreak >nul
goto :mysql_ready_loop

:check_port_available
set "SERVICE_NAME=%~1"
set "PORT=%~2"
set "MAX_ATTEMPTS=30"
set "ATTEMPT=1"
call :log_info "⏳ 检查 !SERVICE_NAME! 端口 !PORT! 是否监听..."

:port_check_loop
if !ATTEMPT! GTR !MAX_ATTEMPTS! (
    call :log_error "!SERVICE_NAME! 端口 !PORT! 检查超时"
)

netstat -an | findstr /R /C:"^  TCP.*:!PORT! .*LISTENING" >nul
if not errorlevel 1 (
    call :log_info "✅ !SERVICE_NAME! 端口 !PORT! 已开放"
    exit /b 0
)

call :log_info "⏳ !SERVICE_NAME! 端口 !PORT! 尚未开放（第 !ATTEMPT!/!MAX_ATTEMPTS! 次尝试）..."
set /a ATTEMPT+=1
timeout /t 5 /nobreak >nul
goto :port_check_loop

:: =============== 主流程 ===============
:main

:: 1. 创建 volume 目录和 mcp_settings.json 文件
call :log_info "📁 创建 volume 目录和配置文件..."
if not exist ".\volume\mcp-data" (
    mkdir ".\volume\mcp-data" 2>nul
    if errorlevel 1 (
        call :log_error "无法创建目录 .\volume\mcp-data"
    )
)

if not exist ".\volume\mcp-data\mcp_settings.json" (
    echo {} > ".\volume\mcp-data\mcp_settings.json"
    if errorlevel 1 (
        call :log_error "无法创建文件 .\volume\mcp-data\mcp_settings.json"
    )
    call :log_info "✅ mcp_settings.json 已初始化为有效 JSON"
)

:: 2. 启动 Docker Compose 服务
call :log_info "🐳 启动 Docker Compose 服务..."
docker-compose up -d
if errorlevel 1 (
    call :log_error "Docker Compose 启动失败，请检查 docker-compose.yml"
)

:: 3. 检查 Python 环境
call :log_info "🔍 检查 Python 环境..."
where python >nul 2>&1
if errorlevel 1 (
    call :log_error "未检测到 Python。请从 https://www.python.org/downloads/ 安装并勾选 'Add to PATH'"
)

pip --version >nul 2>&1
if errorlevel 1 (
    call :log_error "未检测到 pip。请确保 Python 安装完整"
)

for /f "tokens=*" %%v in ('python --version 2^>nul') do set "PYTHON_VERSION=%%v"
call :log_info "✅ Python 环境正常 (!PYTHON_VERSION!)"

:: 4. 安装 Python 依赖
call :log_info "🐍 安装 Python 依赖 (pymysql, py2neo)..."
pip install pymysql py2neo
if errorlevel 1 (
    call :log_error "Python 依赖安装失败"
)

:: 5. 检查服务状态
call :log_info "🔍 检查容器和服务状态..."

call :wait_for_container "chat-db"
set "MYSQL_CONTAINER_OK=!errorlevel!"

call :wait_for_container "neo4j-apoc"
set "NEO4J_CONTAINER_OK=!errorlevel!"

if "!MYSQL_CONTAINER_OK!"=="0" (
    call :check_mysql_ready
    set "MYSQL_READY_OK=!errorlevel!"
) else (
    set "MYSQL_READY_OK=1"
)

call :check_port_available "MySQL" 13006
set "MYSQL_PORT_OK=!errorlevel!"

call :check_port_available "Neo4j" 7687
set "NEO4J_PORT_OK=!errorlevel!"

:: 6. 执行数据库初始化（仅当所有服务就绪）
if "!MYSQL_CONTAINER_OK!"=="0" && "!NEO4J_CONTAINER_OK!"=="0" && "!MYSQL_READY_OK!"=="0" && "!MYSQL_PORT_OK!"=="0" && "!NEO4J_PORT_OK!"=="0" (
    call :log_info "📊 所有服务就绪，等待 15 秒确保稳定..."
    timeout /t 15 /nobreak >nul

    set "MYSQL_INIT=../common/initialize_mysql.py"
    set "NEO4J_INIT=../common/initialize_neo4j.py"

    if not exist "!MYSQL_INIT!" (
        call :log_error "MySQL 初始化脚本不存在: !MYSQL_INIT!"
    )
    if not exist "!NEO4J_INIT!" (
        call :log_error "Neo4j 初始化脚本不存在: !NEO4J_INIT!"
    )

    set "MAX_RETRY=3"
    set "RETRY=1"

    :init_retry
    call :log_info "🔄 第 !RETRY! 次执行数据库初始化..."

    :: 初始化 MySQL
    call :log_info "🗃️  执行 MySQL 表初始化..."
    python "!MYSQL_INIT!"
    if errorlevel 1 (
        call :log_info "⚠️  MySQL 初始化失败"
        goto init_failed
    )

    :: 初始化 Neo4j
    call :log_info "🔗 执行 Neo4j 关系初始化..."
    python "!NEO4J_INIT!"
    if errorlevel 1 (
        call :log_info "⚠️  Neo4j 初始化失败"
        goto init_failed
    )

    call :log_info "🎉 数据库初始化成功完成！"
    goto :success

    :init_failed
    if !RETRY! LSS !MAX_RETRY! (
        set /a RETRY+=1
        call :log_info "⏳ 10 秒后重试初始化..."
        timeout /t 10 /nobreak >nul
        goto :init_retry
    ) else (
        call :log_error "数据库初始化连续 !MAX_RETRY! 次失败，退出部署流程"
    )
) else (
    call :log_error "部分服务未就绪，跳过初始化"
    call :log_info "服务状态汇总:"
    call :log_info "  - MySQL 容器:         !MYSQL_CONTAINER_OK! (0=OK)"
    call :log_info "  - Neo4j 容器:         !NEO4J_CONTAINER_OK!"
    call :log_info "  - MySQL 服务就绪:     !MYSQL_READY_OK!"
    call :log_info "  - MySQL 端口(13006):  !MYSQL_PORT_OK!"
    call :log_info "  - Neo4j 端口(7687):   !NEO4J_PORT_OK!"
    exit /b 1
)

:success
call :log_info "✅ 部署与初始化全流程成功完成！"
goto :end

:end
echo.
echo 按任意键退出...
pause >nul
exit /b 0