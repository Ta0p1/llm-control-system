param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$PullMissingModels
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Get-OllamaCommand {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Test-PythonPackage {
    param([string]$Name)
    python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('$Name') else 1)" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

Write-Host "Checking Python..." -ForegroundColor Cyan
python --version

Write-Host "Checking Ollama..." -ForegroundColor Cyan
$ollamaExe = Get-OllamaCommand
if (-not $ollamaExe) {
    Write-Error "Ollama is not installed or not on PATH."
}

try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 5
} catch {
    Write-Warning "Ollama API is not reachable on http://127.0.0.1:11434 . Start Ollama before using the app."
    $tags = $null
}

$requiredPackages = @("fastapi", "uvicorn", "httpx", "qdrant_client", "langgraph", "pypdf", "pptx", "PIL", "sympy", "scipy", "control")
$missing = @()
foreach ($pkg in $requiredPackages) {
    if (-not (Test-PythonPackage -Name $pkg)) {
        $missing += $pkg
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing Python packages: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host "Install them with:" -ForegroundColor Yellow
    Write-Host "python -m pip install -r requirements.txt"
    exit 1
}

$requiredModels = @("qwen3:8b", "qwen2.5vl:7b", "qwen3-embedding:4b")
$optionalModels = @("bge-m3")
$installedModels = @()
if ($tags -and $tags.models) {
    $installedModels = $tags.models | ForEach-Object { $_.name }
}
$missingModels = @()
foreach ($requiredModel in $requiredModels) {
    $matched = $installedModels | Where-Object { $_ -like "$requiredModel*" }
    if (-not $matched) {
        $missingModels += $requiredModel
    }
}

if ($missingModels.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing Ollama models: $($missingModels -join ', ')" -ForegroundColor Yellow
    if ($PullMissingModels) {
        foreach ($model in $missingModels) {
            Write-Host "Pulling $model ..." -ForegroundColor Cyan
            & $ollamaExe pull $model
        }
    } else {
        Write-Host "You can pull them with:" -ForegroundColor Yellow
        Write-Host ".\pull_models.ps1"
    }
    exit 1
}

$missingOptionalModels = @()
foreach ($optionalModel in $optionalModels) {
    $matched = $installedModels | Where-Object { $_ -like "$optionalModel*" }
    if (-not $matched) {
        $missingOptionalModels += $optionalModel
    }
}

if ($missingOptionalModels.Count -gt 0) {
    Write-Host ""
    Write-Host "Optional local models missing: $($missingOptionalModels -join ', ')" -ForegroundColor Yellow
}

Write-Host "Starting local app at http://$HostAddress`:$Port using qwen3:8b for text and qwen2.5vl:7b for image parsing" -ForegroundColor Green
python -m uvicorn app.server:app --host $HostAddress --port $Port --reload
