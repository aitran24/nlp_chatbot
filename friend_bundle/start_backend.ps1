$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Kill any existing uvicorn/python holding Qdrant lock
Get-Process -Name "uvicorn","python" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# $env:LLM_PROVIDER = "ollama"# $env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# $env:OLLAMA_MODEL = "llama3.1:8b"
$env:ENABLE_NEO4J = "true"
$env:NEO4J_URI = "neo4j://127.0.0.1:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "12345678"
# $env:QDRANT_PATH = Join-Path $root "cache\qdrant_local_bench"

& ".\.venv\Scripts\uvicorn.exe" "backend.main:app" --host 127.0.0.1 --port 8000 --reload
