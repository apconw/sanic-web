# deploy_and_init.ps1 - 自动化部署和初始化脚本（Windows PowerShell 版，含内嵌 init_data 逻辑）

# 兼容 Windows 10/11 的 PowerShell 版本检查
$PSVersion = $PSVersionTable.PSVersion.Major
if ($PSVersion -lt 3) {
    Write-Host "错误: 需要 PowerShell 3.0 或更高版本，当前版本: $PSVersion" -ForegroundColor Red
    exit 1
}

# 使用兼容的 StrictMode 设置
if ($PSVersion -ge 3) {
    Set-StrictMode -Version 3.0 -ErrorAction Stop
} else {
    Set-StrictMode -Off
}

# 设置错误处理
$ErrorActionPreference = "Continue"

# 获取脚本所在目录，确保路径正确
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ScriptDir

$ErrorLog = Join-Path $ScriptDir "error.log"
$DeployLog = Join-Path $ScriptDir "deploy.log"

function Write-Log {
    param (
        [string]$Message,
        [string]$Level = "Info"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "${timestamp}: ${Level} - ${Message}"

    if ($Level -eq "Error") {
        Write-Host "错误: $Message" -ForegroundColor Red
        $logEntry | Out-File -FilePath $ErrorLog -Append -Encoding UTF8
    } else {
        Write-Host $Message
        $logEntry | Out-File -FilePath $DeployLog -Append -Encoding UTF8
    }
}

Write-Log "开始部署和初始化流程..."

# === 第1~5步：创建配置、启动Docker、检查环境等（与之前一致）===

Write-Log "创建volume目录和配置文件..."
$VolumePath = Join-Path $ScriptDir "volume\mcp-data"
if (!(Test-Path -LiteralPath $VolumePath)) {
    try {
        New-Item -ItemType Directory -Path $VolumePath -Force | Out-Null
        if ($LASTEXITCODE -ne 0 -and -not (Test-Path -LiteralPath $VolumePath)) {
            throw "无法创建目录"
        }
    } catch {
        Write-Log "无法创建目录 $VolumePath : $_" -Level Error
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
        "12306-mcp"
      ],
      "owner": "admin"
    },
    "mcp-server-chart": {
      "type": "stdio",
      "command": "npx",
      "args": [
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

Write-Log "启动Docker服务..."
try {
    $dockerComposeFile = Join-Path $ScriptDir "docker-compose.yaml"
    if (Test-Path -LiteralPath $dockerComposeFile) {
        & docker-compose -f $dockerComposeFile up -d
    } else {
        & docker-compose up -d
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Docker服务启动失败，退出代码: $LASTEXITCODE" -Level Error
    }
} catch {
    Write-Log "Docker服务启动异常: $_" -Level Error
}

Write-Log "检查Python环境..."
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
    Write-Log "Python环境检查通过 (版本: $pyVer)"
}

if ($HasPip) {
    Write-Log "安装Python依赖..."
    & pip install pymysql py2neo
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Python依赖安装失败" -Level Error
    }
}

function Wait-Container {
    param([string]$Name, [int]$MaxAttempts = 30)
    Write-Log "等待容器 $Name 启动..."
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        try {
            $state = docker inspect -f "{{.State.Running}}" $Name 2>&1
            if ($LASTEXITCODE -eq 0 -and $state -eq "true") {
                Write-Log "容器 $Name 已成功启动"
                return $true
            }
        } catch {
            # 忽略错误，继续重试
        }
        Write-Log "容器 $Name 尚未启动，第 $attempt/$MaxAttempts 次尝试..."
        Start-Sleep -Seconds 10
        $attempt++
    }
    Write-Log "容器 $Name 启动超时" -Level Error
    return $false
}

function Test-MySqlReady {
    param([int]$MaxAttempts = 30)
    Write-Log "等待 MySQL 服务准备就绪..."
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        try {
            $result = docker exec mysql-db mysqladmin ping --silent 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Log "MySQL 服务已准备就绪"
                return $true
            }
        } catch {
            # 忽略错误，继续重试
        }
        Write-Log "MySQL 尚未准备就绪，第 $attempt/$MaxAttempts 次尝试..."
        Start-Sleep -Seconds 5
        $attempt++
    }
    Write-Log "MySQL 服务准备超时" -Level Error
    return $false
}

function Test-PortOpen {
    param([string]$Service, [int]$Port, [int]$MaxAttempts = 30)
    Write-Log "检查 $Service 端口 $Port 是否可用..."
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        try {
            # 兼容 Windows 10/11 的端口检查
            $conn = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
            if ($conn -and $conn.TcpTestSucceeded) {
                Write-Log "$Service 端口 $Port 已开放"
                return $true
            }
        } catch {
            # 如果 Test-NetConnection 不可用，尝试使用 telnet 或 socket 方式
            try {
                $tcpClient = New-Object System.Net.Sockets.TcpClient
                $tcpClient.ConnectTimeout = 2000
                $tcpClient.Connect("localhost", $Port)
                if ($tcpClient.Connected) {
                    $tcpClient.Close()
                    Write-Log "$Service 端口 $Port 已开放"
                    return $true
                }
            } catch {
                # 继续重试
            }
        }
        Write-Log "$Service 端口 $Port 尚未开放，第 $attempt/$MaxAttempts 次尝试..."
        Start-Sleep -Seconds 5
        $attempt++
    }
    Write-Log "$Service 端口 $Port 检查超时" -Level Error
    return $false
}

function Test-Neo4jReady {
    param([int]$MaxAttempts = 60)
    Write-Log "等待 Neo4j Bolt 服务准备就绪..."
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        # 方法1: 尝试使用 cypher-shell 连接（如果容器内有）
        try {
            $null = docker exec neo4j-apoc cypher-shell -u neo4j -p neo4j123 "RETURN 1;" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Log "Neo4j Bolt 服务已准备就绪"
                return $true
            }
        } catch {
            # 忽略错误，继续尝试其他方法
        }
        
        # 方法2: 尝试使用 Python 脚本测试 Bolt 连接
        if ($HasPython) {
            try {
                $pythonScript = @"
import socket
import sys
try:
    # 首先检查端口是否开放
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', 7687))
    sock.close()
    if result == 0:
        # 端口开放，尝试 Bolt 握手
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock2.settimeout(2)
        sock2.connect(('localhost', 7687))
        # 发送 Bolt 握手消息（Bolt协议版本协商）
        handshake = b'\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        sock2.send(handshake)
        response = sock2.recv(4)
        sock2.close()
        if len(response) >= 4:
            sys.exit(0)
    sys.exit(1)
except Exception as e:
    sys.exit(1)
"@
                $tempScript = Join-Path $env:TEMP "check_neo4j_bolt_$PID.py"
                try {
                    $pythonScript | Out-File -FilePath $tempScript -Encoding UTF8 -ErrorAction Stop
                    $null = & python $tempScript 2>&1
                    $pythonExitCode = $LASTEXITCODE
                    if ($pythonExitCode -eq 0) {
                        Write-Log "Neo4j Bolt 服务已准备就绪"
                        return $true
                    }
                } finally {
                    # 确保清理临时文件
                    if (Test-Path -LiteralPath $tempScript) {
                        Remove-Item $tempScript -ErrorAction SilentlyContinue
                    }
                }
            } catch {
                # 继续重试
            }
        }
        
        # 方法3: 简单的端口连接测试（作为后备方案）
        try {
            $tcpClient = New-Object System.Net.Sockets.TcpClient
            $tcpClient.ReceiveTimeout = 2000
            $tcpClient.SendTimeout = 2000
            $connectResult = $tcpClient.BeginConnect("localhost", 7687, $null, $null)
            $waitResult = $connectResult.AsyncWaitHandle.WaitOne(2000, $false)
            if ($waitResult -and $tcpClient.Connected) {
                $tcpClient.EndConnect($connectResult)
                # 尝试发送 Bolt 握手
                try {
                    $stream = $tcpClient.GetStream()
                    $handshake = [byte[]](0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
                    $stream.Write($handshake, 0, $handshake.Length)
                    $buffer = New-Object byte[] 4
                    $bytesRead = $stream.Read($buffer, 0, 4)
                    $stream.Close()
                    if ($bytesRead -ge 4) {
                        Write-Log "Neo4j Bolt 服务已准备就绪"
                        return $true
                    }
                } catch {
                    # 忽略握手错误
                }
                $tcpClient.Close()
            }
        } catch {
            # 继续重试
        }
        
        Write-Log "Neo4j Bolt 服务尚未准备就绪，第 $attempt/$MaxAttempts 次尝试..."
        Start-Sleep -Seconds 5
        $attempt++
    }
    Write-Log "Neo4j Bolt 服务准备超时" -Level Error
    return $false
}

$container_mysql_ok = Wait-Container "mysql-db"
$container_neo4j_ok = Wait-Container "neo4j-apoc"
$mysql_ready_ok = Test-MySqlReady
$neo4j_ready_ok = Test-Neo4jReady
$port_mysql_ok = Test-PortOpen "MySQL" 13006
$port_neo4j_ok = Test-PortOpen "Neo4j" 7687

# === 内嵌 init_data.sh 的逻辑（不再调用外部脚本）===

if ($container_mysql_ok -and $container_neo4j_ok -and $mysql_ready_ok -and $neo4j_ready_ok -and $port_mysql_ok -and $port_neo4j_ok) {
    Write-Log "等待服务稳定 (10秒)..."
    Start-Sleep -Seconds 10

    # 检查 SQL 文件是否存在（使用绝对路径）
    $SqlFile = Join-Path $ScriptDir "init_sql.sql"
    if (!(Test-Path -LiteralPath $SqlFile)) {
        Write-Log "Error: SQL file '$SqlFile' not found." -Level Error
        exit 1
    }

    # 获取项目根目录（脚本目录的父目录）
    $ProjectRoot = Split-Path -Parent $ScriptDir
    
    # 执行 initialize_mysql.py（使用绝对路径）
    $MysqlScript = Join-Path $ProjectRoot "common\initialize_mysql.py"
    if (Test-Path -LiteralPath $MysqlScript) {
        Write-Log "执行 MySQL 初始化脚本..."
        try {
            # 切换到项目根目录执行，确保相对路径正确
            Push-Location $ProjectRoot
            & python $MysqlScript
            $mysqlExitCode = $LASTEXITCODE
            Pop-Location
            if ($mysqlExitCode -ne 0) {
                Write-Log "MySQL 初始化失败，退出代码: $mysqlExitCode" -Level Error
                exit 1
            }
        } catch {
            Pop-Location
            Write-Log "MySQL 初始化异常: $_" -Level Error
            exit 1
        }
    } else {
        Write-Log "MySQL 初始化脚本未找到: $MysqlScript" -Level Error
        exit 1
    }

    # 执行 initialize_neo4j.py（使用绝对路径）
    $Neo4jScript = Join-Path $ProjectRoot "common\initialize_neo4j.py"
    if (Test-Path -LiteralPath $Neo4jScript) {
        Write-Log "执行 Neo4j 初始化脚本..."
        $neo4jMaxRetries = 3
        $neo4jRetryCount = 0
        $neo4jSuccess = $false
        
        while ($neo4jRetryCount -lt $neo4jMaxRetries -and -not $neo4jSuccess) {
            try {
                # 切换到项目根目录执行，确保相对路径正确
                Push-Location $ProjectRoot
                & python $Neo4jScript
                $neo4jExitCode = $LASTEXITCODE
                Pop-Location
                
                if ($neo4jExitCode -eq 0) {
                    $neo4jSuccess = $true
                    Write-Log "Neo4j 初始化成功"
                } else {
                    $neo4jRetryCount++
                    if ($neo4jRetryCount -lt $neo4jMaxRetries) {
                        Write-Log "Neo4j 初始化失败，退出代码: $neo4jExitCode，等待 10 秒后重试 ($neo4jRetryCount/$neo4jMaxRetries)..." -Level Error
                        Start-Sleep -Seconds 10
                        # 再次检查 Neo4j 是否就绪
                        if (-not (Test-Neo4jReady -MaxAttempts 10)) {
                            Write-Log "Neo4j 服务似乎不可用，继续重试..." -Level Error
                        }
                    } else {
                        Write-Log "Neo4j 初始化失败，已重试 $neo4jMaxRetries 次，退出代码: $neo4jExitCode" -Level Error
                        exit 1
                    }
                }
            } catch {
                Pop-Location
                $neo4jRetryCount++
                if ($neo4jRetryCount -lt $neo4jMaxRetries) {
                    Write-Log "Neo4j 初始化异常: $_，等待 10 秒后重试 ($neo4jRetryCount/$neo4jMaxRetries)..." -Level Error
                    Start-Sleep -Seconds 10
                } else {
                    Write-Log "Neo4j 初始化异常，已重试 $neo4jMaxRetries 次: $_" -Level Error
                    exit 1
                }
            }
        }
        
        if (-not $neo4jSuccess) {
            Write-Log "Neo4j 初始化最终失败" -Level Error
            exit 1
        }
    } else {
        Write-Log "Neo4j 初始化脚本未找到: $Neo4jScript" -Level Error
        exit 1
    }

    Write-Log "部署和初始化完成！"
    exit 0
} else {
    Write-Log "服务启动失败，无法执行数据初始化" -Level Error
    Write-Log "各服务状态:"
    Write-Log "- MySQL容器启动: $container_mysql_ok"
    Write-Log "- Neo4j容器启动: $container_neo4j_ok"
    Write-Log "- MySQL服务就绪: $mysql_ready_ok"
    Write-Log "- Neo4j Bolt服务就绪: $neo4j_ready_ok"
    Write-Log "- MySQL端口可用: $port_mysql_ok"
    Write-Log "- Neo4j端口可用: $port_neo4j_ok"
    exit 1
}