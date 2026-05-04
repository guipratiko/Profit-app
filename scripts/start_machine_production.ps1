$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$containerName = "profit-app-backend"
$tunnelName = "profit-app-backend"
$dockerDesktopPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
$cloudflaredConfigPath = "C:\Users\Aiko\.cloudflared\config.yml"
$logDir = Join-Path $repoRoot "storage\logs"
$stdoutLogPath = Join-Path $logDir "cloudflared.stdout.log"
$stderrLogPath = Join-Path $logDir "cloudflared.stderr.log"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

if (-not (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue)) {
    if (Test-Path $dockerDesktopPath) {
        Start-Process -FilePath $dockerDesktopPath | Out-Null
    }
}

$dockerReady = $false
$deadline = (Get-Date).AddMinutes(4)
while ((Get-Date) -lt $deadline) {
    try {
        docker info | Out-Null
        $dockerReady = $true
        break
    } catch {
        Start-Sleep -Seconds 5
    }
}

if (-not $dockerReady) {
    throw "Docker engine was not available before timeout."
}

$containerStarted = $false
try {
    docker start $containerName | Out-Null
    $containerStarted = $true
} catch {
    $containerStarted = $false
}

if (-not $containerStarted) {
    Push-Location $repoRoot
    try {
        docker compose up -d | Out-Null
    } finally {
        Pop-Location
    }
}

if (-not (Test-Path $cloudflaredConfigPath)) {
    throw "cloudflared config file not found at $cloudflaredConfigPath"
}

$cloudflaredRunning = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "cloudflared.exe" -and $_.CommandLine -match "tunnel" -and $_.CommandLine -match $tunnelName
    }

if (-not $cloudflaredRunning) {
    Start-Process -FilePath "cloudflared" -ArgumentList @("tunnel", "--config", $cloudflaredConfigPath, "run", $tunnelName) -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutLogPath -RedirectStandardError $stderrLogPath | Out-Null
}