# DUT RAG Backend

Backend này là lớp phục vụ cho hệ thống hỏi đáp của project, tách phần `query -> retrieve -> rerank -> answer` ra khỏi notebook để có thể:

- chạy như API qua `FastAPI`
- query dữ liệu thật từ `Qdrant`
- kết hợp `Neo4j` cho câu hỏi có cấu trúc
- dùng `reranker` để lọc lại các chunk liên quan nhất
- đổi LLM linh hoạt giữa `Groq` và `Ollama local`

## Backend đang làm gì

Luồng xử lý hiện tại:

1. Nhận câu hỏi từ user
2. Parse entity và intent
3. Quyết định route:
   - `vector_only`
   - `hybrid` (`Neo4j + Qdrant`)
4. Lấy candidate chunks từ `Qdrant`
5. Re-rank candidate bằng `CrossEncoder`
6. Gộp graph facts + text chunks thành context
7. Gọi LLM để sinh câu trả lời

Các loại entity backend đang nhận biết:

- `MSSV`
- `tên sinh viên`
- `phòng`
- `ngày`
- `ann_id`
- `đợt`
- `khóa`
- `môn`
- `TOEIC / TOEFL / VSTEP / GDQP`

Các kiểu route chính:

- câu hỏi quy định chung: vector search
- câu hỏi có `MSSV`: graph trước, vector sau
- câu hỏi theo `phòng / môn / khóa / đợt`: graph lọc bản ghi, rồi vector lấy ngữ cảnh
- câu hỏi có `ann_id`: filter thẳng theo thông báo

## File chính

- [main.py](/d:/ki8/xlnntn/243/backend/main.py): `FastAPI` app và các endpoint
- [rag_service.py](/d:/ki8/xlnntn/243/backend/rag_service.py): toàn bộ logic retrieval, rerank, graph lookup, generation
- [config.py](/d:/ki8/xlnntn/243/backend/config.py): cấu hình qua biến môi trường
- [requirements.txt](/d:/ki8/xlnntn/243/backend/requirements.txt): dependency backend

## API

Endpoint hiện có:

- `GET /health`
- `POST /plan`
- `POST /query`

### `GET /health`

Trả trạng thái cơ bản:

- backend có lên không
- `Neo4j` có bật không
- `reranker` có bật không

### `POST /plan`

Dùng để xem backend sẽ retrieve như thế nào trước khi gọi LLM.

Ví dụ body:

```json
{
  "query": "Mã số sinh viên 102220255 có trong danh sách tốt nghiệp 2025 không?",
  "top_k": 5
}
```

Kết quả trả về gồm:

- `entities`
- `plan`
- `graph_match`
- `vector_preview`

### `POST /query`

Chạy full pipeline và trả:

- `answer`
- `context`
- `entities`
- `plan`
- `graph_match`
- `vector_hits`

## Retriever và reranker

Phần này nằm trong [rag_service.py](/d:/ki8/xlnntn/243/backend/rag_service.py).

Các bước:

- `retrieve_vector(...)`: lấy candidate từ `Qdrant`
- `rerank_hits(...)`: chấm lại candidate bằng `CrossEncoder`
- `build_context(...)`: cắt context theo `MAX_CTX_CHARS`

Config hiện tại:

- `RERANKER_MODEL = cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- `RERANK_CANDIDATES = 12`
- `RERANK_TOP_K = 5`

Nghĩa là hiện tại backend đang làm kiểu:

- lấy khoảng `12` candidate đầu
- rerank lại
- giữ `top 5`

Nếu muốn khớp đúng slide kiểu `top 10 -> top 3`, chỉ cần sửa:

- `RERANK_CANDIDATES=10`
- `RERANK_TOP_K=3`

## LLM provider

Backend hỗ trợ 2 kiểu sinh câu trả lời:

### 1. Groq

Dùng khi bạn muốn gọi model hosted.

Biến môi trường:

```bash
set LLM_PROVIDER=groq
set GROQ_API_KEY=...
set GROQ_MODEL=llama-3.1-8b-instant
```

### 2. Ollama local

Dùng khi bạn muốn chạy model local trên máy.

Biến môi trường:

```bash
set LLM_PROVIDER=ollama
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set OLLAMA_MODEL=llama3.1:8b
```

Hiện tại project đã test được với:

- `llama3.1:8b`
- `qwen2.5:7b`

## Dữ liệu backend đang dùng

### Qdrant

Nguồn vector mặc định:

- `data_sv1/qdrant_local`

Khi benchmark/test để tránh lock file, có thể dùng:

- `data_sv1/qdrant_local_test`

### Neo4j

Graph hiện được dùng cho các query cấu trúc.

Schema chính đang thấy trong data:

- `SinhVien`
- `Announcement`
- `KiThi`
- `Record`

Notebook ingest graph liên quan:

- [notebooks/xulicsv.ipynb](</d:/ki8/xlnntn/243/notebooks/xulicsv.ipynb>)

## Cách chạy backend

### Chạy với Ollama local

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="llama3.1:8b"
$env:ENABLE_NEO4J="true"
$env:NEO4J_URI="neo4j://127.0.0.1:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="12345678"
$env:QDRANT_PATH="d:\ki8\xlnntn\243\data_sv1\qdrant_local_test"
.\.venv\Scripts\uvicorn.exe backend.main:app --reload
```

### Chạy với Groq

```powershell
$env:LLM_PROVIDER="groq"
$env:GROQ_API_KEY="..."
$env:GROQ_MODEL="llama-3.1-8b-instant"
$env:ENABLE_NEO4J="true"
$env:NEO4J_URI="neo4j://127.0.0.1:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="12345678"
$env:QDRANT_PATH="d:\ki8\xlnntn\243\data_sv1\qdrant_local_test"
.\.venv\Scripts\uvicorn.exe backend.main:app --reload
```

## Benchmark

Script benchmark:

- [scripts/run_clean_benchmark.py](/d:/ki8/xlnntn/243/scripts/run_clean_benchmark.py)

Script vẽ đồ thị:

- [scripts/plot_benchmark_results.py](/d:/ki8/xlnntn/243/scripts/plot_benchmark_results.py)

### Benchmark local cùng cỡ

Ví dụ benchmark `llama3.1:8b` vs `qwen2.5:7b`:

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:ENABLE_NEO4J="false"
$env:ENABLE_RERANKER="true"
$env:RERANKER_LOCAL_ONLY="true"
$env:QDRANT_PATH="d:\ki8\xlnntn\243\data_sv1\qdrant_local_test"
.\.venv\Scripts\python.exe scripts\run_clean_benchmark.py `
  --provider ollama `
  --cache-path data_sv1\inference_results_local_same_size.json `
  --output-json data_sv1\inference_results_local_same_size.json `
  --output-csv data_sv1\eval_results_local_same_size.csv `
  --summary-json data_sv1\eval_summary_local_same_size.json `
  --top-k 5 `
  --temperature 0.0 `
  --max-tokens 160 `
  --main-model llama3.1:8b `
  --baseline-model qwen2.5:7b `
  --rerun-main all `
  --rerun-baseline
```

### Vẽ đồ thị từ kết quả benchmark

```powershell
.\.venv\Scripts\python.exe scripts\plot_benchmark_results.py `
  --csv data_sv1\eval_results_local_same_size.csv `
  --summary data_sv1\eval_summary_local_same_size.json `
  --out-dir data_sv1\plots_local_same_size
```

Các đồ thị được sinh:

- `overall_metrics.png`
- `bertscore_by_doc_type.png`
- `bertscore_distribution.png`
- `validity_pies.png`

## Ghi chú thực tế

- `Neo4j` chỉ mạnh khi dữ liệu tương ứng đã được ingest vào graph.
- `Qdrant` local thường bị lock nếu có nhiều process cùng mở, nên benchmark hay dùng bản `_test`.
- Với benchmark local bằng `Ollama`, tốc độ sẽ chậm hơn Groq nhưng ổn định hơn và không bị quota `429`.
- Hiện backend core đã test được qua gọi trực tiếp `RagService().answer(...)`, nhưng benchmark full 150 câu local trước đó đã bị dừng giữa chừng nên cần chạy lại đến hết để có kết quả cuối.
