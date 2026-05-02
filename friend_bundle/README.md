# DUT Chatbot Share Bundle

Gói này đủ để một máy khác chạy:

- `FE` chat giao diện web
- `BE` FastAPI backend
- `Qdrant local` đã embed sẵn
- nguồn `CSV/Excel` để import `Neo4j`

## Cấu trúc

- `backend/`: API và RAG pipeline
- `FE/`: frontend chat
- `cache/qdrant_local_bench/`: vector DB local đã nhúng sẵn
- `graph_data/csv_01a/`: danh sách thi dạng CSV
- `graph_data/excel/`: file Excel dùng để nạp graph
- `scripts/import_neo4j.py`: script import graph riêng
- `start_backend.ps1`: chạy backend
- `start_fe.ps1`: chạy frontend
- `start_all.ps1`: chạy cả FE + BE

## Yêu cầu

- Windows + PowerShell
- Python 3.11+ hoặc tương đương
- Neo4j Desktop hoặc Neo4j local
- Ollama

## Chuẩn bị môi trường

Trong thư mục `friend_bundle`:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Cài model local:

```powershell
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

## Import graph vào Neo4j

1. Mở Neo4j local.
2. Bảo đảm URI và password đúng với script.
3. Chạy import:

```powershell
.\.venv\Scripts\python.exe scripts\import_neo4j.py --reset
```

Nếu password khác mặc định:

```powershell
$env:NEO4J_PASSWORD="mat_khau_cua_ban"
.\.venv\Scripts\python.exe scripts\import_neo4j.py --reset
```

Mặc định script đọc:

- `graph_data/csv_01a`
- `graph_data/excel`

Script này được tách ra từ logic trong notebook `xulicsv.ipynb`.

## Chạy hệ thống

```powershell
.\start_all.ps1
```

Mở:

- Frontend: `http://127.0.0.1:4173`
- Backend health: `http://127.0.0.1:8000/health`

## Chạy riêng từng phần

Backend:

```powershell
.\start_backend.ps1
```

Frontend:

```powershell
.\start_fe.ps1
```

## Ghi chú

- `Qdrant` trong bundle đã có sẵn dữ liệu embed, không cần chạy lại chunking/embedding.
- `Neo4j` cần import một lần đầu để các query dạng danh sách, MSSV, phòng thi, đợt thi hoạt động đúng.
- `start_backend.ps1` đang mặc định:
  - `OLLAMA_MODEL=llama3.1:8b`
  - `ENABLE_NEO4J=true`
  - `NEO4J_PASSWORD=12345678`
- Nếu máy khác dùng password khác, sửa trực tiếp `start_backend.ps1` hoặc set env trước khi chạy.
