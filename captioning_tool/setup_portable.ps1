param([switch]$RuntimeOnly)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$runtime = Join-Path $root 'runtime'
$pythonDir = Join-Path $runtime 'python'
$python = Join-Path $pythonDir 'python.exe'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$env:PIP_CACHE_DIR = Join-Path $runtime 'pip-cache'
$env:PYTHONNOUSERSITE = '1'
function Download($url, $destination) {
    $partial = $destination + '.partial'
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $partial -TimeoutSec 120
    Move-Item -LiteralPath $partial -Destination $destination -Force
}
if (-not (Test-Path -LiteralPath $python)) {
    $archive = Join-Path $runtime 'python-3.12.10-embed-amd64.zip'
    Write-Host 'Downloading portable Python 3.12.10 (Windows x64)...'
    Download 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $pythonDir -Force
}
# Relative search paths keep the embedded interpreter relocatable.
@('python312.zip', '.', 'Lib\site-packages', '..\..', 'import site') | Set-Content -LiteralPath (Join-Path $pythonDir 'python312._pth') -Encoding ascii
if (-not (Test-Path -LiteralPath (Join-Path $pythonDir 'Lib\site-packages\pip\__main__.py'))) {
    $getPip = Join-Path $runtime 'get-pip.py'
    Download 'https://bootstrap.pypa.io/get-pip.py' $getPip
    & $python $getPip --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed; rerun setup_portable.ps1' }
}
if ($RuntimeOnly) { exit 0 }
Write-Host 'Installing CUDA PyTorch and captioning dependencies locally...'
& $python -m pip install --only-binary=:all: 'torch==2.12.0' 'torchvision==0.27.0' --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $python -m pip install --only-binary=:all: -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $python -c "import torch; from transformers import Qwen3VLForConditionalGeneration, LlavaForConditionalGeneration, Sam2Model; print('CUDA available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Dependency import check failed' }
Set-Content -LiteralPath (Join-Path $runtime 'ready.txt') -Value 'portable-v1' -Encoding ascii
Write-Host 'Setup complete. Models download on the first run. NVIDIA driver is supplied by the host PC.'
