Set-Location $PSScriptRoot
uv run python scripts/run_project.py @args
exit $LASTEXITCODE
