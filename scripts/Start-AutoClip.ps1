$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DockerExe = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $DockerExe) {
    $DockerExe = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
}
if (-not (Test-Path -LiteralPath $DockerExe)) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show('Docker Desktop is not installed. Install it before starting AutoClip.', 'AutoClip') | Out-Null
    exit 1
}

$desktop = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
if (-not (Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue) -and (Test-Path -LiteralPath $desktop)) {
    Start-Process -FilePath $desktop -WindowStyle Hidden
}

$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    & $DockerExe info *> $null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ready) { throw 'Docker did not become ready within 120 seconds.' }

Push-Location $ProjectDir
try {
    $running = & $DockerExe compose ps --services --status running 2>$null
    if ($running -notcontains 'autoclip') {
        & $DockerExe compose up -d --build
        if ($LASTEXITCODE -ne 0) { throw 'AutoClip failed to start.' }
    }
} finally {
    Pop-Location
}
Start-Process 'http://localhost:8000'
