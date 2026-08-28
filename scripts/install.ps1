$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ProjectSpec = $ProjectRoot + "[dev]"

& $PythonBin -m venv $VenvDir
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install $ProjectSpec

Write-Host "Installed P BrainZ into $VenvDir"
Write-Host "The system Python was not modified."
Write-Host "Next: run .\scripts\run.ps1, then configure providers in the native control panel"
