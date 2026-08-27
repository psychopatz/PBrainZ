$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (!(Test-Path -LiteralPath $VenvPython)) {
    if (!(Get-Command $PythonBin -ErrorAction SilentlyContinue)) {
        throw "Python 3.11 or newer is required. Install Python, then run this script again."
    }

    Write-Host "First run: creating the private HoomansLLM environment..."
    & $PythonBin -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the HoomansLLM virtual environment."
    }
}

$Probe = "import fastapi, google.genai, openai, pydantic_settings, uvicorn, hoomans_llm"
$null = & $VenvPython -c $Probe 2>$null
$NeedsDependencies = $LASTEXITCODE -ne 0

if ($NeedsDependencies) {
    Write-Host "First run: installing HoomansLLM dependencies..."
    & $VenvPython -m ensurepip --upgrade *> $null
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Could not prepare pip inside the HoomansLLM virtual environment."
    }
    & $VenvPython -m pip install $ProjectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install HoomansLLM dependencies."
    }
}

Set-Location -LiteralPath $ProjectRoot
& $VenvPython -m hoomans_llm @args
exit $LASTEXITCODE
