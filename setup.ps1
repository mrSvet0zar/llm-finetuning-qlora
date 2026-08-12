# ============================================================
#  Setup environnement — Windows / PowerShell
#  Cree un venv, installe PyTorch (CUDA 12.4) puis les deps.
# ============================================================
#  Usage :  .\setup.ps1
# ------------------------------------------------------------

$ErrorActionPreference = "Stop"

Write-Host "==> Creation de l'environnement virtuel (.venv)" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "==> Activation" -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

Write-Host "==> Mise a jour de pip" -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "==> Installation de PyTorch (CUDA 12.4)" -ForegroundColor Cyan
pip install torch --index-url https://download.pytorch.org/whl/cu124

Write-Host "==> Installation des dependances du projet" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "==> Verification GPU / CUDA" -ForegroundColor Cyan
python -c "import torch; print('CUDA disponible :', torch.cuda.is_available()); print('GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'aucun')"

Write-Host ""
Write-Host "Setup termine. Etapes suivantes :" -ForegroundColor Green
Write-Host "  python scripts/generate_dataset.py   # generer le dataset brut"
Write-Host "  python prepare_dataset.py            # valider + splitter"
Write-Host "  python train.py                      # entrainer (QLoRA)"
