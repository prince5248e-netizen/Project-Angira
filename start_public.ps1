$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Create it with: python -m venv .venv"
}

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw "cloudflared is not installed or not on PATH. Install it from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
}

$server = Start-Process -FilePath $python `
    -ArgumentList "app.py" `
    -WorkingDirectory $root `
    -PassThru

try {
    Start-Sleep -Seconds 3
    Write-Host "Voice Authenticity Checker is running locally at http://localhost:5000"
    Write-Host "Starting a public HTTPS tunnel. Keep this window open."
    & cloudflared tunnel --url http://localhost:5000
}
finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id
    }
}
