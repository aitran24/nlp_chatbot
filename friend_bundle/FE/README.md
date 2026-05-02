# DUT Chat Frontend

Frontend tĩnh cho backend RAG hiện tại.

## File

- `index.html`: layout chat
- `styles.css`: giao diện
- `app.js`: gọi API `GET /health`, `GET /models`, `POST /plan`, `POST /query`

## Chạy

Từ root project:

```powershell
.\start_all.ps1
```

Hoặc chạy riêng:

```powershell
.\start_backend.ps1
.\start_fe.ps1
```

Mở:

- `http://127.0.0.1:4173`

Backend API:

- `http://127.0.0.1:8000/health`
