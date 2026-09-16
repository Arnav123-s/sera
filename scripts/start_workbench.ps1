param([int]$Port = 8765)
$projectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectPath
& "$projectPath/.venv/Scripts/python.exe" -X utf8 -m workbench --port $Port
