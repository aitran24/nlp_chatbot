$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

& ".\.venv\Scripts\python.exe" -m http.server 4173 --bind 127.0.0.1 --directory FE
