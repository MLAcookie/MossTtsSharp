# download MossTTS ONNX Models
# (targetDir)
# ├ MOSS-TTS-Nano-100M-ONNX
# └ MOSS-Audio-Tokenizer-Nano-ONNX
param(
    [string]$TargetDir = "MossTtsModel",
    [switch]$Force
)
$ModelUrl = "https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
$TokenizerUrl = "https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"

function Invoke-GitClone {
    param(
        [string]$CloneUrl
    )
    Write-Host "Cloning $CloneUrl ..." -ForegroundColor Cyan
    $CloneArgs = @(
        "clone",
        "--depth=1",
        $CloneUrl
    )
    $Result = & git @CloneArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: git clone failed." -ForegroundColor Red
        Write-Host $Result -ForegroundColor Red 
        exit $LASTEXITCODE
    }
    Write-Host ""
}

Write-Host "=== MossTTS Model Downloader ===" -ForegroundColor Cyan

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Error: git is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

$target = Join-Path (Get-Location) $TargetDir

if (Test-Path -LiteralPath $target) {
    if ($Force) {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
    }
    else {
        Write-Host "Warning: '$TargetDir' already exists." -ForegroundColor Yellow
        Write-Host "[C]onfirm [Q]uit" -ForegroundColor Yellow
        $choice = Read-Host
        switch ($choice.ToLowerInvariant()) {
            'c' { Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue }
            default { Write-Host "Cancelled." -ForegroundColor Gray; exit 0 }
        }
    }
}

New-Item $target -ItemType Directory -ErrorAction SilentlyContinue | Out-Null

Push-Location $target
try {
    Invoke-GitClone $ModelUrl
    Invoke-GitClone $TokenizerUrl
}
finally {
    Pop-Location
}

Write-Host "Done. Models downloaded to: $target" -ForegroundColor Green
