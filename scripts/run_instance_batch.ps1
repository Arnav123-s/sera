$ErrorActionPreference = 'Stop'
$seraPython = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
$seraRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $seraRoot
foreach ($seraSeed in @(307, 311, 313, 317, 331)) {
    foreach ($seraArm in @('observed_only', 'corrected_teacher', 'uncorrected_teacher', 'oracle_exposure')) {
        $seraJob = "runs/v3-batch-001/A06-INSTANCE-$seraSeed-$seraArm"
        if (Test-Path -LiteralPath $seraJob) { throw "Existing owned cell: $seraJob" }
        & $seraPython -X utf8 scripts/run_instance_bounded.py --seconds 150 --output $seraJob --module experiments.instance_transfer.study -- run --arm $seraArm --seed $seraSeed
        if ($LASTEXITCODE -ne 0) { throw "Owned instance-transfer cell failed: $seraJob" }
    }
}
