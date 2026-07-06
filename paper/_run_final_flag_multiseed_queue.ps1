$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Results = Join-Path $Root "paper\results"
$Logs = Join-Path $Results "logs"
$Python = "D:\Desktop\SWC-Studio\.venv\Scripts\python.exe"
$StatusLog = Join-Path $Logs "final_flag_multiseed_queue.status.log"

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Stamp {
    return (Get-Date).ToString("s")
}

function Write-Status {
    param([string]$Message)
    $line = "[$(Stamp)] $Message"
    Add-Content -Path $StatusLog -Value $line
    Write-Host $line
}

function Test-NonEmptyFile {
    param([string]$Path)
    return (Test-Path $Path) -and ((Get-Item $Path).Length -gt 100)
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string]$LogName,
        [string[]]$Arguments
    )
    $logPath = Join-Path $Logs $LogName
    Write-Status "$Name start -> $logPath"
    Push-Location $Root
    try {
        & $Python -u @Arguments 2>&1 | Tee-Object -FilePath $logPath
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
    Write-Status "$Name complete"
}

if (!(Test-Path $Python)) {
    throw "Missing Python venv: $Python"
}

Write-Status "final flag multiseed queue start"

$Seeds = @(123, 42, 789)
foreach ($seed in $Seeds) {
    $labels = Join-Path $Results "final_flag_seed${seed}_labels.csv"
    $features = Join-Path $Results "final_flag_seed${seed}_features.csv"
    $summary = Join-Path $Results "final_flag_seed${seed}_summary.json"
    if ((Test-NonEmptyFile $labels) -and (Test-NonEmptyFile $features) -and (Test-NonEmptyFile $summary)) {
        Write-Status "seed $seed flag scoring skipped; existing outputs found"
    }
    else {
        Invoke-PythonStep `
            -Name "seed $seed score flag labels/features" `
            -LogName "final_flag_seed${seed}_score.log" `
            -Arguments @(
                "-m", "paper._score_model_split_for_flag",
                "--model-dir", "paper/models/v12_gentle_seed${seed}",
                "--split", "test",
                "--progress-every", "100",
                "--out-labels", "paper/results/final_flag_seed${seed}_labels.csv",
                "--out-features", "paper/results/final_flag_seed${seed}_features.csv",
                "--out-summary", "paper/results/final_flag_seed${seed}_summary.json"
            )
    }
}

Invoke-PythonStep `
    -Name "combine multiseed flag rows and leakage precheck" `
    -LogName "final_flag_multiseed_combine.log" `
    -Arguments @(
        "-m", "paper._combine_final_flag_multiseed",
        "--out-labels", "paper/results/final_flag_multiseed_labels.csv",
        "--out-features", "paper/results/final_flag_multiseed_features.csv",
        "--out-check", "paper/results/final_flag_multiseed_leakage_check.json"
    )

$heavyFeatures = Join-Path $Results "final_flag_multiseed_features_oof_heavy.csv"
$heavySummary = Join-Path $Results "final_flag_multiseed_features_oof_heavy.summary.json"
if ((Test-NonEmptyFile $heavyFeatures) -and (Test-NonEmptyFile $heavySummary)) {
    Write-Status "out-of-fold heavy feature build skipped; existing outputs found"
}
else {
    Invoke-PythonStep `
        -Name "build out-of-fold baseline and v12 disagreement features" `
        -LogName "final_flag_oof_heavy_features.log" `
        -Arguments @(
            "-m", "paper._build_final_flag_oof_features",
            "--labels", "paper/results/final_flag_multiseed_labels.csv",
            "--base-features", "paper/results/final_flag_multiseed_features.csv",
            "--out-features", "paper/results/final_flag_multiseed_features_oof_heavy.csv",
            "--out-heavy", "paper/results/final_flag_multiseed_oof_heavy_features_only.csv",
            "--progress-every", "100"
        )
}

Invoke-PythonStep `
    -Name "train/evaluate no-leak leave-one-seed-out flag models" `
    -LogName "final_flag_loso_train_eval.log" `
    -Arguments @(
        "-m", "paper._train_eval_final_flag_multiseed",
        "--labels", "paper/results/final_flag_multiseed_labels.csv",
        "--features", "paper/results/final_flag_multiseed_features_oof_heavy.csv",
        "--out-csv", "paper/results/final_flag_leave_one_seed_out.csv",
        "--out-summary", "paper/results/final_flag_leave_one_seed_out_summary.csv",
        "--out-before-after", "paper/results/final_flag_before_after.csv",
        "--out-json", "paper/results/final_flag_leave_one_seed_out.json",
        "--model-dir", "paper/models/final_flag_multiseed_loso",
        "--scopes", "all,pyramidal,interneuron",
        "--feature-modes", "compact,branch3,baseline_oof,v12_oof"
    )

Write-Status "final flag multiseed queue complete"
