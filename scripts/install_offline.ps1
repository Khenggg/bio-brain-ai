$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $Root ".venv"
$WheelsDir = Join-Path $Root "wheels"
$LockFile = Join-Path $Root "requirements.lock.txt"

$UvCmd = Get-Command uv -ErrorAction SilentlyContinue
$UvFallback = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
$UvExe = if ($UvCmd) { $UvCmd.Source } elseif (Test-Path $UvFallback) { $UvFallback } else { $null }

$PythonExe = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null)
}
if (-not $PythonExe -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $PythonExe = (Get-Command python).Source
}

if ($UvExe) {
    if (-not (Test-Path $VenvDir)) {
        Write-Host "Creating virtual environment at $VenvDir with uv..."
        if ($PythonExe) {
            & $UvExe venv $VenvDir --python $PythonExe
        } else {
            & $UvExe venv $VenvDir --python 3.12
        }
    }
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    Write-Host "Installing dependencies offline from $WheelsDir..."
    & $UvExe pip install --python $VenvPython --no-index --find-links $WheelsDir -r $LockFile
} else {
    if (-not $PythonExe) {
        throw "Neither uv nor Python 3.12 was found on PATH."
    }
    if (-not (Test-Path $VenvDir)) {
        Write-Host "Creating virtual environment at $VenvDir with python -m venv..."
        & $PythonExe -m venv $VenvDir
    }
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    Write-Host "Installing dependencies offline from $WheelsDir with pip..."
    & $VenvPython -m pip install --no-index --find-links $WheelsDir -r $LockFile
}

Write-Host "Verifying PyTorch CUDA..."
& $VenvPython -c "import torch, cv2, faiss, transformers; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
