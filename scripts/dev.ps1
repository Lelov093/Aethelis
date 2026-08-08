[CmdletBinding()]
param(
    [switch]$SkipDatabaseCheck,
    [switch]$ExitAfterReady,
    [ValidateRange(5, 120)]
    [int]$StartupTimeoutSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $repositoryRoot "frontend"
$logRoot = Join-Path $repositoryRoot "tmp\dev"

function Resolve-Executable([string]$name) {
    foreach ($candidate in @($name, "$name.cmd", "$name.exe")) {
        $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $command) {
            return [string]$command.Path
        }
    }
    return $null
}

function Get-ListeningProcess([int]$port) {
    return Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Start-LoggedProcess(
    [string]$name,
    [string]$filePath,
    [string[]]$argumentList
) {
    $standardOutputPath = Join-Path $logRoot "$name.out.log"
    $standardErrorPath = Join-Path $logRoot "$name.err.log"
    Remove-Item -LiteralPath $standardOutputPath, $standardErrorPath -Force -ErrorAction SilentlyContinue

    $process = Start-Process `
        -FilePath $filePath `
        -ArgumentList $argumentList `
        -WorkingDirectory $repositoryRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $standardOutputPath `
        -RedirectStandardError $standardErrorPath `
        -PassThru

    return [pscustomobject]@{
        Name = $name
        Process = $process
        StandardOutputPath = $standardOutputPath
        StandardErrorPath = $standardErrorPath
    }
}

function Stop-LoggedProcess($entry) {
    if ($null -eq $entry -or $null -eq $entry.Process) {
        return
    }
    $entry.Process.Refresh()
    if ($entry.Process.HasExited) {
        return
    }
    try {
        $taskkillExecutable = Resolve-Executable "taskkill"
        if ($taskkillExecutable) {
            & $taskkillExecutable /PID $entry.Process.Id /T /F *> $null
        }
        else {
            Stop-Process -Id $entry.Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    catch {
        Stop-Process -Id $entry.Process.Id -Force -ErrorAction SilentlyContinue
    }
}

function Show-ProcessLogs($entry) {
    foreach ($logPath in @($entry.StandardOutputPath, $entry.StandardErrorPath)) {
        if (Test-Path -LiteralPath $logPath) {
            $lines = Get-Content -LiteralPath $logPath -Tail 30 -ErrorAction SilentlyContinue
            if ($lines) {
                Write-Host "[$($entry.Name)] $logPath" -ForegroundColor Yellow
                $lines | Write-Host
            }
        }
    }
}

function Assert-ProcessesRunning($entries) {
    foreach ($entry in $entries) {
        $entry.Process.Refresh()
        if ($entry.Process.HasExited) {
            Show-ProcessLogs $entry
            throw "Development process '$($entry.Name)' exited with code $($entry.Process.ExitCode)."
        }
    }
}

function Wait-ForEndpoint(
    [string]$name,
    [string]$uri,
    $entries,
    [int]$timeoutSeconds
) {
    $deadline = [DateTimeOffset]::Now.AddSeconds($timeoutSeconds)
    do {
        Assert-ProcessesRunning $entries
        try {
            $response = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Write-Host "$name ready: $uri" -ForegroundColor Green
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 250
        }
    } while ([DateTimeOffset]::Now -lt $deadline)

    throw "$name did not become ready within $timeoutSeconds seconds: $uri"
}

$uvExecutable = Resolve-Executable "uv"
$pnpmExecutable = Resolve-Executable "pnpm"
$pnpmPrefix = @()
if (-not $pnpmExecutable) {
    $corepackExecutable = Resolve-Executable "corepack"
    if ($corepackExecutable) {
        $pnpmExecutable = $corepackExecutable
        $pnpmPrefix = @("pnpm")
    }
}
if (-not $uvExecutable) {
    throw "Required command 'uv' is not available. Install uv or add it to this PowerShell PATH."
}
if (-not $pnpmExecutable) {
    throw "Required command 'pnpm' is not available. Install pnpm, or run 'corepack enable' and reopen PowerShell."
}

if (-not (Test-Path (Join-Path $frontendRoot "node_modules"))) {
    throw "Frontend dependencies are missing. Run: corepack pnpm --dir frontend install --frozen-lockfile"
}

if (-not $SkipDatabaseCheck) {
    Push-Location $repositoryRoot
    try {
        & $uvExecutable run alembic current
        if ($LASTEXITCODE -ne 0) {
            throw "PostgreSQL/Alembic check failed. The script does not install or delete PostgreSQL."
        }
    }
    finally {
        Pop-Location
    }
}

foreach ($port in @(8000, 5173)) {
    $listener = Get-ListeningProcess $port
    if ($listener) {
        throw "Port $port is already in use by PID $($listener.OwningProcess). Stop that process before starting Aethelis."
    }
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$processes = @()
$frontendArguments = @($pnpmPrefix) + @("--dir", "frontend", "dev")

Write-Host "Aethelis local stack is starting." -ForegroundColor Cyan
Write-Host "Frontend package manager: $pnpmExecutable $($pnpmPrefix -join ' ')" -ForegroundColor DarkGray
Write-Host "Logs: $logRoot" -ForegroundColor DarkGray

try {
    $processes += Start-LoggedProcess "api" $uvExecutable @("run", "aethelis-api")
    $processes += Start-LoggedProcess "worker" $uvExecutable @("run", "aethelis-worker")
    $processes += Start-LoggedProcess "frontend" $pnpmExecutable $frontendArguments

    Wait-ForEndpoint "Product API" "http://127.0.0.1:8000/healthz" $processes $StartupTimeoutSeconds
    Wait-ForEndpoint "Player client" "http://localhost:5173" $processes $StartupTimeoutSeconds

    Write-Host "Aethelis local stack is ready." -ForegroundColor Cyan
    Write-Host "Player client: http://localhost:5173" -ForegroundColor Green
    Write-Host "Product API:   http://127.0.0.1:8000" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop all three processes." -ForegroundColor DarkGray

    if ($ExitAfterReady) {
        Write-Host "Startup validation completed; stopping validation processes." -ForegroundColor DarkGray
        return
    }

    while ($true) {
        Assert-ProcessesRunning $processes
        Start-Sleep -Milliseconds 500
    }
}
catch {
    foreach ($entry in $processes) {
        Show-ProcessLogs $entry
    }
    throw
}
finally {
    foreach ($entry in $processes) {
        Stop-LoggedProcess $entry
    }
}
