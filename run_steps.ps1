param()
$ErrorActionPreference = "Stop"

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Host ".venv not found."
    exit 1
}

Write-Host "============================"
Write-Host "Step 1: Verify the active Python interpreter"
python --version

Write-Host "`n============================"
Write-Host "Step 2: Verify which pip is being used"
where.exe pip

Write-Host "`n============================"
Write-Host "Step 3: Upgrade pip inside the virtual environment"
python -m pip install --upgrade pip

Write-Host "`n============================"
Write-Host "Step 4: Install the required dependencies inside the environment"
python -m pip install torch torchvision pillow numpy matplotlib opencv-python scipy

Write-Host "`n============================"
Write-Host "Step 5: Verify the installation by opening Python and running:"
python -c "import torch; import torchvision; from PIL import Image; print('PyTorch version:', torch.__version__)"

Write-Host "`n============================"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Step 6: Confirmation - All imports succeeded."
} else {
    Write-Host "Step 6: Import verification failed."
}
