$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$dataset = Join-Path $root 'datasets\meow_v02_sft_v2_1_semantic_verified'
$state = Join-Path $dataset 'semantic_judge_run.json'
$log = Join-Path $dataset 'automatic_finalize.log'
while (-not (Test-Path -LiteralPath $state)) {
    Start-Sleep -Seconds 30
}
do {
    $run = Get-Content -LiteralPath $state -Raw | ConvertFrom-Json
    if ($run.status -eq 'COMPLETED') { break }
    Start-Sleep -Seconds 30
} while ($true)
Push-Location $root
try {
    "[$(Get-Date -Format o)] semantic judge complete; finalizing" | Add-Content -LiteralPath $log
    & python scripts\sft_semantic_verified_v2_1.py finalize *>> $log
    & python scripts\validate_sft_semantic_verified_v2_1.py *>> $log
    "[$(Get-Date -Format o)] finalize and validation complete" | Add-Content -LiteralPath $log
} finally {
    Pop-Location
}
