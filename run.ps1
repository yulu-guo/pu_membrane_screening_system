$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$Candidates = @(
    (Join-Path $HOME "anaconda3\envs\xiao\python.exe"),
    (Join-Path $HOME "miniconda3\envs\xiao\python.exe")
)
$Python = $null
foreach ($Candidate in $Candidates) {
    if (Test-Path -LiteralPath $Candidate) {
        $Python = $Candidate
        break
    }
}
if (-not $Python) {
    $Python = "python"
}
$env:PYTHONIOENCODING = "utf-8"
Set-Location -LiteralPath $ProjectDirectory
& $Python "app.py" --host "127.0.0.1" --port 8766
