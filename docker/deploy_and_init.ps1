# deploy_and_init.ps1 - 自动化部署和初始化脚本（Windows PowerShell 版，含内嵌 init_data 逻辑）

Set-StrictMode -Version Latest

$ErrorLog = "error.log"
$DeployLog = "deploy.log"

function Write-Log {
    param (
        [string]$Message,
        [string]$Level = "Info"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "$timestamp: $Level - $Message"

    if ($Level -eq "Error") {
        Write-Host "❌ 错误: $Message" -ForegroundColor Red
        $logEntry | Out-File -FilePath $ErrorLog -Append -Encoding UTF8
    } else {
        Write-Host $Message
        $logEntry | Out-File -FilePath $DeployLog -Append -Encoding UTF8
    }
}

Write-Log "🚀 开始部署和初始化流程..."

# === 第1~5步：创建配置、启动Docker、检查环境等（与之前一致）===

Write-Log "📁 创建volume目录和配置文件..."
$VolumePath = "volume\mcp-data"
if (!(Test-Path $VolumePath)) {
    try {
        New-Item -ItemType Directory -Path $VolumePath -Force | Out-Null
    } catch {
        Write-Log "无法创建目录 $VolumePath" -Level Error
    }
}

$ConfigFile = Join-Path $VolumePath "mcp_settings.json"
$JsonContent = @'
{
  "mcpServers": {
    "12306": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "12306-mcp"
      ],
      "owner": "admin"
    },
    "mcp-server-chart": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@antv/mcp-server-chart"
      ],
      "env": {
        "VIS_REQUEST_SERVER": "http://gpt-vis-api:3000/generate"
      },
      "owner": "admin"
    }
  },
  "users": [
    {
      "username": "admin",
      "password": "$2b$10$dZBmE4IAFBy1fOFUJ9itZekn1eX3WzS1i1thI.gl9LBh9tukmtk1W",
      "isAdmin": true
    }
  ],
  "groups": [
    {
      "id": "d7af20c7-1b08-4963-82b6-41affc54a20d",
      "name": "common-qa",
      "description": "",
      "servers": [
        {
          "name": "12306",
          "tools": "all"
        },
        {
          "name": "amap",
          "tools": "all"
        },
        {
          "name": "mcp-server-firecrawl",
          "tools": "all"
        },
        {
          "name": "mcp-server-chart",
          "tools": "all"
        }
      ],
      "owner": "admin"
    },
    {
      "id": "71a21b11-d684-462d-9005-79bc62934d88",
      "name": "database-qa",
      "description": "",
      "servers": [
        {
          "name": "mcp-server-chart",
          "tools": "all"
        }
      ],
      "owner": "admin"
    }
  ],
  "systemConfig": {
    "routing": {
      "enableGlobalRoute": true,
      "enableGroupNameRoute": true,
      "enableBearerAuth": false,
      "bearerAuthKey": "TnGDRZ4bHlnOA5mKqoG5CSonSepsI798",
      "skipAuth": false
    },
    "install": {
      "pythonIndexUrl": "https://mirrors.aliyun.com/pypi/simple",
      "npmRegistry": "https://registry.npmmirror.com",
      "baseUrl": "http://localhost:3300"
    },
    "smartRouting": {
      "enabled": false,
      "dbUrl": "",
      "openaiApiBaseUrl": "",
      "openaiApiKey": "",
      "openaiApiEmbeddingModel": ""
    },
    "mcpRouter": {
      "apiKey": "",
      "referer": "https://mcphub.app",
      "title": "MCPHub",
      "baseUrl": "https://api.mcprouter.to/v1"
    }
  }
}
'@

try {
    Set-Content -Path $ConfigFile -Value $JsonContent -Encoding UTF8
} catch {
    Write-Log "无法创建文件 $ConfigFile" -Level Error
}

Write-Log "🐳 启动Docker服务..."
& docker-compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Log "Docker服务启动失败" -Level Error
}

Write-Log "🔍 检查Python环境..."
$HasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
$HasPip = $null -ne (Get-Command pip -ErrorAction SilentlyContinue)

if (-not $HasPython) {
    Write-Log "未检测到Python环境" -Level Error
    Write-Log "请从 https://www.python.org/downloads/ 安装 Python 并勾选 'Add to PATH'"
}

if (-not $HasPip) {
    Write-Log "未检测到pip工具" -Level Error
    Write-Log "可运行: python -m ensurepip --upgrade"
}

if ($HasPython) {
    $pyVer = & python --version 2>&1
    Write-Log "✅ Python环境检查通过 (版本: $pyVer)"
}

if ($HasPip) {
    Write-Log "🐍 安装Python依赖..."
    & pip install pymysql py2neo
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Python依赖安装失败" -Level Error
    }
}

function Wait-Container {
    param([string]$Name, [int]$MaxAttempts = 30)
    Write-Log "⏳ 等待容器 $Name 启动..."
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        $state = docker inspect -f "{{.State.Running}}" $Name 2>$null
        if ($state -eq "true") {
            Write-Log "✅ 容器 $Name 已成功启动"
            return $true
        }
        Write-Log "⏳ 容器 $Name 尚未启动，第 $attempt/$MaxAttempts 次尝试..."
        Start-Sleep -Seconds 10
        $attempt++
    }
    Write-Log "容器 $Name 启动超时" -Level Error
    return $false
}

function Test-MySqlReady {
    param([int]$MaxAttempts = 30)
    Write-Log "⏳ 等待 MySQL 服务准备就绪..."
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        $result = docker exec mysql-db mysqladmin ping --silent 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "✅ MySQL 服务已准备就绪"
            return $true
        }
        Write-Log "⏳ MySQL 尚未准备就绪，第 $attempt/$MaxAttempts 次尝试..."
        Start-Sleep -Seconds 5
        $attempt++
    }
    Write-Log "MySQL 服务准备超时" -Level Error
    return $false
}

function Test-PortOpen {
    param([string]$Service, [int]$Port, [int]$MaxAttempts = 30)
    Write-Log "⏳ 检查 $Service 端口 $Port 是否可用..."
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        $conn = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
        if ($conn.TcpTestSucceeded) {
            Write-Log "✅ $Service 端口 $Port 已开放"
            return $true
        }
        Write-Log "⏳ $Service 端口 $Port 尚未开放，第 $attempt/$MaxAttempts 次尝试..."
        Start-Sleep -Seconds 5
        $attempt++
    }
    Write-Log "$Service 端口 $Port 检查超时" -Level Error
    return $false
}

$container_mysql_ok = Wait-Container "mysql-db"
$container_neo4j_ok = Wait-Container "neo4j-apoc"
$mysql_ready_ok = Test-MySqlReady
$port_mysql_ok = Test-PortOpen "MySQL" 13006
$port_neo4j_ok = Test-PortOpen "Neo4j" 7687

# === 内嵌 init_data.sh 的逻辑（不再调用外部脚本）===

if ($container_mysql_ok -and $container_neo4j_ok -and $mysql_ready_ok -and $port_mysql_ok -and $port_neo4j_ok) {
    Write-Log "📊 等待服务稳定 (30秒)..."
    Start-Sleep -Seconds 30

    # 检查 SQL 文件是否存在（相对路径）
    $SqlFile = "init_sql.sql"
    if (!(Test-Path $SqlFile)) {
        Write-Log "Error: SQL file '$SqlFile' not found." -Level Error
        exit 1
    }

    # 执行 initialize_mysql.py
    $MysqlScript = "..\common\initialize_mysql.py"
    if (Test-Path $MysqlScript) {
        Write-Log "🔧 执行 MySQL 初始化脚本..."
        & python $MysqlScript
        if ($LASTEXITCODE -ne 0) {
            Write-Log "MySQL 初始化失败" -Level Error
            exit 1
        }
    } else {
        Write-Log "MySQL 初始化脚本未找到: $MysqlScript" -Level Error
        exit 1
    }

    # 执行 initialize_neo4j.py
    $Neo4jScript = "..\common\initialize_neo4j.py"
    if (Test-Path $Neo4jScript) {
        Write-Log "🔧 执行 Neo4j 初始化脚本..."
        & python $Neo4jScript
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Neo4j 初始化失败" -Level Error
            exit 1
        }
    } else {
        Write-Log "Neo4j 初始化脚本未找到: $Neo4jScript" -Level Error
        exit 1
    }

    Write-Log "🎉 部署和初始化完成！"
} else {
    Write-Log "服务启动失败，无法执行数据初始化" -Level Error
    Write-Log "各服务状态:"
    Write-Log "- MySQL容器启动: $container_mysql_ok"
    Write-Log "- Neo4j容器启动: $container_neo4j_ok"
    Write-Log "- MySQL服务就绪: $mysql_ready_ok"
    Write-Log "- MySQL端口可用: $port_mysql_ok"
    Write-Log "- Neo4j端口可用: $port_neo4j_ok"
}