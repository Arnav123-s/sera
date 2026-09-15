$ErrorActionPreference = 'Stop'
$seraPython = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
$seraRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $seraRoot
$seraCandidates = @('A06-TRANSFER-001', 'A06-TRANSFER-002')
$seraSeeds = @(181, 191, 211, 223, 227)
$seraArms = @('observed_only', 'detached', 'integrated', 'extra_capacity', 'oracle_exposure')
foreach ($seraCandidate in $seraCandidates) {
    foreach ($seraSeed in $seraSeeds) {
        foreach ($seraArm in $seraArms) {
            $seraJob = "runs/v3-batch-001/$seraCandidate-$seraSeed-$seraArm"
            # Never silently restart, skip, or overwrite a partial/completed cell.
            if (Test-Path -LiteralPath $seraJob) { throw "Existing owned cell: $seraJob" }
            & $seraPython -X utf8 scripts/run_transfer_bounded.py --seconds 150 --output $seraJob --module experiments.cross_route_transfer.study -- run --name $seraCandidate --arm $seraArm --seed $seraSeed
            if ($LASTEXITCODE -ne 0) { throw "Owned research cell failed: $seraJob" }
        }
    }
}
