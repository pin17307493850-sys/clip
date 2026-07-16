$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DockerExe = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $DockerExe) { $DockerExe = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe' }
if (-not (Test-Path -LiteralPath $DockerExe)) { exit 0 }
& $DockerExe info *> $null
if ($LASTEXITCODE -ne 0) { exit 0 }
Push-Location $ProjectDir
try {
    $containers = & $DockerExe compose ps -q 2>$null
    if ($containers) { & $DockerExe compose down }
} finally {
    Pop-Location
}
