param(
    [switch]$SkipLargeModel,
    [switch]$SkipVisionModel
)

$ErrorActionPreference = "Stop"

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

$ollamaExe = Get-OllamaCommand
if (-not $ollamaExe) {
    Write-Error "Ollama is not installed or not on PATH."
}

$models = @("qwen3.5:9b", "qwen3-embedding:4b", "bge-m3")

foreach ($model in $models) {
    Write-Host "Pulling $model ..." -ForegroundColor Cyan
    & $ollamaExe pull $model
}

Write-Host "Done. You can now run .\\start.ps1" -ForegroundColor Green
