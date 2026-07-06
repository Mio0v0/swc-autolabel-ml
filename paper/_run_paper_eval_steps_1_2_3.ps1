param(
    [int[]]$EvalSeeds = @(42, 789),
    [int[]]$AblationSeeds = @(123, 42, 789),
    [switch]$ForceV12,
    [switch]$ForceAblation,
    [switch]$ForceBaselines
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = "D:\Desktop\SWC-Studio\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$LogDir = Join-Path $Root "paper\results\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$StatusLog = Join-Path $LogDir "paper_eval_steps_1_2_3.status.log"

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format s), $Message
    Add-Content -Path $StatusLog -Value $line
    Write-Host $line
}

function Run-Logged {
    param(
        [string]$Name,
        [string]$LogPath,
        [scriptblock]$Block
    )
    Write-Status "$Name start -> $LogPath"
    & $Block *>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        Write-Status "$Name FAILED rc=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Status "$Name complete"
}

function Assert-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Missing expected output: $Path"
    }
    $null = Get-Content $Path -Raw | ConvertFrom-Json
}

function Assert-Baselines {
    param([int]$Seed)
    $path = Join-Path $Root "paper\results\baselines_on_v12_seed$Seed.json"
    Assert-JsonFile $path
    $payload = Get-Content $path -Raw | ConvertFrom-Json
    if ($payload.seed -ne $Seed) {
        throw "Baseline seed mismatch for seed $Seed in $path"
    }
    if ($payload.reports.Count -ne 4) {
        throw "Expected 4 baseline reports for seed $Seed, found $($payload.reports.Count)"
    }
}

Write-Status "paper eval steps 1/2/3 start; eval seeds=$($EvalSeeds -join ','); ablation seeds=$($AblationSeeds -join ',')"

# Preserve the canonical seed123 baseline result under the seed-specific name used by the compiler.
$seed123Baseline = Join-Path $Root "paper\results\baselines_on_v12.json"
if (Test-Path $seed123Baseline) {
    $payload = Get-Content $seed123Baseline -Raw | ConvertFrom-Json
    if ($payload.seed -eq 123) {
        Copy-Item -Force $seed123Baseline (Join-Path $Root "paper\results\baselines_on_v12_seed123.json")
        foreach ($ext in @("csv", "txt")) {
            $src = Join-Path $Root "paper\results\baselines_on_v12.$ext"
            if (Test-Path $src) {
                Copy-Item -Force $src (Join-Path $Root "paper\results\baselines_on_v12_seed123.$ext")
            }
        }
        $srcTable = Join-Path $Root "paper\results\baselines_on_v12_table.txt"
        if (Test-Path $srcTable) {
            Copy-Item -Force $srcTable (Join-Path $Root "paper\results\baselines_on_v12_seed123_table.txt")
        }
        Write-Status "seed 123 baseline snapshot refreshed"
    }
}

foreach ($seed in $EvalSeeds) {
    $v12Json = Join-Path $Root "paper\results\v12_gt_celltype_seed$seed.json"
    if ($ForceV12 -or -not (Test-Path $v12Json)) {
        Run-Logged "seed $seed v12+Branch3 GT-celltype eval" `
            (Join-Path $LogDir "paper_eval_seed${seed}_v12_gt_celltype.log") `
            { & $Python -u -m paper._eval_v12_gt_celltype --seed $seed }
    } else {
        Write-Status "seed $seed v12+Branch3 GT-celltype eval skipped; existing $v12Json"
    }
    Assert-JsonFile $v12Json

    $baselineReady = $false
    if (-not $ForceBaselines) {
        try {
            Assert-Baselines $seed
            $baselineReady = $true
            Write-Status "seed $seed baseline retrain/eval skipped; existing seed-specific baseline is complete"
        } catch {
            $baselineReady = $false
        }
    }
    if (-not $baselineReady) {
        $env:SWCAL_MODEL_DIR_SUFFIX = "_seed$seed"
        try {
            Run-Logged "seed $seed baseline retrain/eval" `
                (Join-Path $LogDir "paper_eval_seed${seed}_baselines.log") `
                { & $Python -u -m paper._eval_baselines_on_v12 --seed $seed --force --force-retrain }
        } finally {
            Remove-Item Env:\SWCAL_MODEL_DIR_SUFFIX -ErrorAction SilentlyContinue
        }
    }
    Assert-Baselines $seed

    Run-Logged "seed $seed leakage check" `
        (Join-Path $LogDir "paper_eval_seed${seed}_leakage.log") `
        { & $Python -u -m paper._check_split_leakage --seed $seed --baseline-json "paper\results\baselines_on_v12_seed$seed.json" }

    Run-Logged "seed $seed compare v12 vs baselines" `
        (Join-Path $LogDir "paper_eval_seed${seed}_compare.log") `
        { & $Python -u -m paper._compare_v12_seed_to_baselines --seed $seed --baseline-json "paper\results\baselines_on_v12_seed$seed.json" }
}

foreach ($seed in $AblationSeeds) {
    $ablationJson = Join-Path $Root "paper\results\v12_gt_celltype_seed${seed}_no_branch3.json"
    if ($ForceAblation -or -not (Test-Path $ablationJson)) {
        Run-Logged "seed $seed v12 GT-celltype Branch3-off ablation" `
            (Join-Path $LogDir "paper_eval_seed${seed}_v12_no_branch3.log") `
            { & $Python -u -m paper._eval_v12_gt_celltype --seed $seed --no-branch3 --out-suffix _no_branch3 }
    } else {
        Write-Status "seed $seed Branch3-off ablation skipped; existing $ablationJson"
    }
    Assert-JsonFile $ablationJson
}

Run-Logged "seed 123 leakage check" `
    (Join-Path $LogDir "paper_eval_seed123_leakage.log") `
    { & $Python -u -m paper._check_split_leakage --seed 123 --baseline-json "paper\results\baselines_on_v12_seed123.json" }

Run-Logged "seed 123 compare v12 vs baselines" `
    (Join-Path $LogDir "paper_eval_seed123_compare.log") `
    { & $Python -u -m paper._compare_v12_seed_to_baselines --seed 123 --baseline-json "paper\results\baselines_on_v12_seed123.json" }

Run-Logged "compile multiseed paper tables" `
    (Join-Path $LogDir "paper_eval_compile_multiseed.log") `
    { & $Python -u -m paper._compile_multiseed_paper_tables --seeds "123,42,789" }

Write-Status "paper eval steps 1/2/3 complete"
