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
        "C:\Users\Allen Huang\AppData\Local\Programs\Ollama\ollama.exe",
        "C:\Program Files\Ollama\ollama.exe"
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

$models = @("qwen3:8b", "qwen3-embedding:4b", "bge-m3")
if (-not $SkipLargeModel) {
    $models += "qwen3:14b"
}
if (-not $SkipVisionModel) {
    $models += "qwen2.5vl:3b"
}

foreach ($model in $models) {
    Write-Host "Pulling $model ..." -ForegroundColor Cyan
    & $ollamaExe pull $model
}

Write-Host "Done. You can now run .\\start.ps1" -ForegroundColor Green
