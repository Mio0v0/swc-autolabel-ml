param(
    [int[]]$Seeds = @(42, 789),
    [string]$Python = "D:\Desktop\SWC-Studio\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location $Root

$LogDir = Join-Path $Root "paper\results\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:PYTHONUNBUFFERED = "1"

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "s"), $Message
    $line | Tee-Object -FilePath (Join-Path $LogDir "final_multiseed_training.status.log") -Append
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}

Write-Status "final multi-seed training start; seeds=$($Seeds -join ',')"
foreach ($seed in $Seeds) {
    $modelDir = "paper\models\v12_gentle_seed$seed"
    $baseLog = Join-Path $LogDir "final_multiseed_train_seed$seed.log"
    $branch3Log = Join-Path $LogDir "final_multiseed_branch3_seed$seed.log"

    Write-Status "seed $seed base v12 training start -> $baseLog"
    & $Python -u -m paper._retrain_v12_gentle_seed --seed $seed *>&1 |
        Tee-Object -FilePath $baseLog
    if ($LASTEXITCODE -ne 0) {
        Write-Status "seed $seed base v12 training FAILED rc=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Status "seed $seed base v12 training complete"

    Write-Status "seed $seed Branch3 training start -> $branch3Log"
    & $Python -u -m paper.gnn_branch3_rescue `
        --model-dir $modelDir `
        --ckpt "$modelDir\gnn_branch3_rescue.pt" `
        --seed $seed *>&1 |
        Tee-Object -FilePath $branch3Log
    if ($LASTEXITCODE -ne 0) {
        Write-Status "seed $seed Branch3 training FAILED rc=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Status "seed $seed Branch3 training complete"
}
Write-Status "final multi-seed training complete"
