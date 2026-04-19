param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Launching desktop app..." -ForegroundColor Cyan
python -m pip install -r requirements.txt
python desktop_app.py
