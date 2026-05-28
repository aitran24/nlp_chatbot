"""
Script thêm dữ liệu từ Excel (danh sách SV) vào dut_chunks Qdrant collection.
Chạy khi backend đang DỪNG để không bị lock Qdrant.

Usage:
    .\.venv\Scripts\python.exe scripts\index_excel_to_qdrant.py
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

ROOT_DIR = Path(__file__).resolve().parents[1]
EXCEL_DIR = ROOT_DIR / "graph_data" / "excel"
QDRANT_PATH = ROOT_DIR / "cache" / "qdrant_local_bench"
COLLECTION = "dut_chunks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 50  # số SV mỗi chunk text

# Mapping ann_id → (title, doc_type) cho các file Excel muốn index
EXCEL_META: dict[str, tuple[str, str]] = {
    "ANN_1D371CB2BD": ("Danh sách sinh viên giao Đồ án tốt nghiệp kỳ 2 năm học 2025-2026 (V2)", "do_an_tot_nghiep"),
    "ANN_C2F7BCC111": ("Danh sách sinh viên giao Đồ án tốt nghiệp kỳ 2 năm học 2025-2026 (V3)", "do_an_tot_nghiep"),
    "ANN_A66E88D914": ("Danh sách sinh viên nhận đề tài DATN kỳ 2 năm học 2025-2026 (V4)", "do_an_tot_nghiep"),
    "ANN_ECFA9D95A5": ("Danh sách sinh viên nhận đề tài DATN kỳ 2 năm học 2025-2026 (V1)", "do_an_tot_nghiep"),
    "ANN_2E5F94A628": ("Danh sách sinh viên nộp tiền nộp trú ký túc xá học kỳ II năm học 2025-2026", "ky_tuc_xa"),
    "ANN_B53B596634": ("Danh sách sinh viên dự kiến xét tốt nghiệp đợt 1 năm 2026 (V2)", "xet_tot_nghiep"),
    "ANN_D89F7F7A92": ("Danh sách sinh viên dự kiến xét tốt nghiệp đợt 1 năm 2026 (V3)", "xet_tot_nghiep"),
    "ANN_E9BF08EA15": ("Danh sách sinh viên dự kiến xét tốt nghiệp đợt 1 năm 2026 (V1)", "xet_tot_nghiep"),
    "ANN_9E11F72C4E": ("Danh sách sinh viên dự kiến tốt nghiệp tháng 12 năm 2025 (V1)", "xet_tot_nghiep"),
    "ANN_BACAB23551": ("Danh sách sinh viên tốt nghiệp tháng 12 năm 2025", "xet_tot_nghiep"),
    "ANN_47C34DDF11": ("Danh sách sinh viên thời học và cảnh báo học vụ HK2 2024-2025", "canh_bao_hoc_vu"),
    "ANN_C3AF778C7B": ("Danh sách sinh viên thời học và cảnh báo học vụ HK2 2025-2026 (2510)", "canh_bao_hoc_vu"),
    "ANN_8F6F9E7DB7": ("Danh sách sinh viên khá 25b trả học phí HK1 qua HK2 năm học 2025-2026", "hoc_phi"),
    "ANN_443E937C04": ("Danh sách sinh viên hoàn trả học phí năm học 2024-2025", "hoc_phi"),
    "ANN_B4624B0244": ("Danh sách sinh viên hoàn trả học phí HK1 2025-2026 (khóa 20, khóa 25)", "hoc_phi"),
    "ANN_7C3A741E63": ("Quyết định miễn học tiếng Anh/Pháp K2025 HK2 (465 SV)", "mien_hoc_ngoai_ngu"),
    "ANN_BE1800B4EA": ("Danh sách sinh viên đăng ký chốt K41", "dang_ky"),
    "ANN_947D54D31B": ("Danh sách sinh viên GDQP đợt 1 K2025", "gdqp"),
}


def find_header_row(df_raw: pd.DataFrame) -> int:
    for idx, row in df_raw.iterrows():
        non_empty = [str(v).strip() for v in row if str(v).strip() not in ("", "nan", "NaN")]
        if len(non_empty) >= 3:
            return int(idx)
    return 0


def read_excel(path: Path) -> pd.DataFrame | None:
    try:
        raw = pd.read_excel(path, header=None, nrows=15)
        hrow = find_header_row(raw)
        df = pd.read_excel(path, header=hrow)
        df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
        df = df.dropna(how="all").ffill().astype(str).replace("nan", "").replace("NaN", "")
        return df
    except Exception as e:
        print(f"  [skip] {path.name}: {e}")
        return None


def pick_name_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["họ và tên", "hoten", "ho ten", "họ tên", "ho_ten", "hò và tên", "họvàtên"]):
            return c
    return None


def pick_id_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["số thẻ", "so the", "sothesv", "mã số", "mssv", "ma so", "mahs"]):
            return c
    return None


def pick_class_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["tenlop", "ten lop", "lớp", "lop", "malop", "ma lop", "lớp sh"]):
            return c
    return None


def build_chunks(df: pd.DataFrame, ann_id: str, title: str, doc_type: str, chunk_size: int) -> list[dict]:
    name_col = pick_name_col(df)
    id_col = pick_id_col(df)
    class_col = pick_class_col(df)

    rows_text = []
    for _, row in df.iterrows():
        parts = []
        if id_col:
            sv_id = str(row.get(id_col, "")).strip()
            if sv_id and sv_id not in ("nan", ""):
                parts.append(f"({sv_id})")
        if name_col:
            name = str(row.get(name_col, "")).strip()
            if name and name not in ("nan", ""):
                parts.insert(0, name)
        if class_col:
            cls = str(row.get(class_col, "")).strip()
            if cls and cls not in ("nan", ""):
                parts.append(f"lớp {cls}")
        if parts:
            rows_text.append(", ".join(parts))

    if not rows_text:
        return []

    chunks = []
    total = len(rows_text)
    for i in range(0, total, chunk_size):
        batch = rows_text[i: i + chunk_size]
        batch_no = i // chunk_size + 1
        num_batches = (total + chunk_size - 1) // chunk_size
        header = f"{title} (phần {batch_no}/{num_batches}, tổng {total} sinh viên):\n"
        text = header + "\n".join(f"- {r}" for r in batch)
        chunks.append({
            "chunk_id": f"{ann_id}_excel_{batch_no}",
            "chunk_type": "excel_list",
            "text": text,
            "ann_id": ann_id,
            "title": title,
            "doc_type": doc_type,
            "ngay_iso": "",
        })
    return chunks


def main():
    print(f"Loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL, local_files_only=True)

    print(f"Opening Qdrant at {QDRANT_PATH}")
    q = QdrantClient(path=str(QDRANT_PATH))

    files = list(EXCEL_DIR.glob("*.xlsx")) + list(EXCEL_DIR.glob("*.xls"))
    total_chunks = 0

    for path in sorted(files):
        ann_match = re.search(r"(ANN_[A-Z0-9]+)", path.name)
        if not ann_match:
            continue
        ann_id = ann_match.group(1)
        if ann_id not in EXCEL_META:
            continue

        title, doc_type = EXCEL_META[ann_id]
        print(f"\n[{ann_id}] {path.name}")
        df = read_excel(path)
        if df is None or df.empty:
            continue

        chunks = build_chunks(df, ann_id, title, doc_type, CHUNK_SIZE)
        if not chunks:
            print(f"  -> Không tìm thấy cột tên/mssv")
            continue

        print(f"  -> {len(df)} rows → {len(chunks)} chunks")

        # Embed và upsert
        texts = [c["text"] for c in chunks]
        vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

        # Xóa chunk cũ của ann_id này (nếu có) trước khi insert
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        q.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="ann_id", match=MatchValue(value=ann_id)),
                      FieldCondition(key="chunk_type", match=MatchValue(value="excel_list"))]
            ),
        )

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=chunk,
            )
            for vec, chunk in zip(vectors, chunks)
        ]
        q.upsert(collection_name=COLLECTION, points=points)
        total_chunks += len(chunks)
        print(f"  -> Inserted {len(chunks)} chunks vào {COLLECTION}")

    q.close()
    print(f"\n[done] Tổng cộng {total_chunks} chunks đã thêm vào {COLLECTION}")


if __name__ == "__main__":
    main()
