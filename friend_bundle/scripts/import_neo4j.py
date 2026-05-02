from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
GRAPH_DATA_DIR = ROOT_DIR / "graph_data"
CSV_DIR = GRAPH_DATA_DIR / "csv_01a"
EXCEL_DIR = GRAPH_DATA_DIR / "excel"

MSSV_COLS = {"mssv", "mã sv", "ma sv", "masv", "mã sinh viên", "student id"}

CYPHER_CSV = """
MERGE (ann:Announcement {id: $ann_id})
  SET ann.title = $title

MERGE (sv:SinhVien {mssv: $msv})
  ON CREATE SET sv.ho_ten = $ho_ten, sv.ngay_sinh = $ngay_sinh

MERGE (ki:KiThi {ki_thi_id: $ki_thi_id})
  ON CREATE SET ki.ann_id = $ann_id,
                ki.loai_thi = $loai_thi,
                ki.ngay_thi = $ngay_thi,
                ki.gio_thi = $gio_thi,
                ki.phong_thi = $phong_thi,
                ki.title = $title,
                ki.dia_diem = $dia_diem

MERGE (sv)-[:THAM_GIA_THI]->(ki)
MERGE (ki)-[:THUOC_TB]->(ann)
"""

CYPHER_RECORD = """
MERGE (ann:Announcement {id: $ann_id})
  SET ann.file_nguon = $file_name
CREATE (r:Record $props)
MERGE (r)-[:THUOC_TB]->(ann)
"""

CYPHER_RECORD_STUDENT = """
MERGE (sv:SinhVien {mssv: $mssv})
WITH sv
MATCH (ann:Announcement {id: $ann_id})
MERGE (sv)-[:CO_TRONG_TB]->(ann)
"""


class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **params):
        with self.driver.session() as session:
            return list(session.run(query, **params))

    def reset(self) -> None:
        self.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self) -> None:
        stmts = [
            "CREATE CONSTRAINT sv_mssv IF NOT EXISTS FOR (n:SinhVien) REQUIRE n.mssv IS UNIQUE",
            "CREATE CONSTRAINT ann_id IF NOT EXISTS FOR (n:Announcement) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT kithi_id IF NOT EXISTS FOR (n:KiThi) REQUIRE n.ki_thi_id IS UNIQUE",
        ]
        for stmt in stmts:
            self.run(stmt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import CSV/Excel graph data into Neo4j.")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "12345678"))
    parser.add_argument("--csv-dir", default=str(CSV_DIR))
    parser.add_argument("--excel-dir", default=str(EXCEL_DIR))
    parser.add_argument("--reset", action="store_true", help="Delete all existing nodes before import.")
    return parser.parse_args()


def find_header_row(path: Path, max_scan: int = 12) -> int:
    raw = pd.read_excel(path, header=None, nrows=max_scan)
    for idx, row in raw.iterrows():
        non_empty = row.dropna().astype(str).str.strip()
        non_empty = non_empty[(non_empty != "nan") & (non_empty != "")]
        if len(non_empty) >= 3:
            return int(idx)
    return 0


def clean_excel_df(path: Path) -> pd.DataFrame | None:
    try:
        header_row = find_header_row(path)
        df = pd.read_excel(path, header=header_row)
        df.columns = [str(col).replace("\n", " ").strip() for col in df.columns]
        df = df.dropna(how="all").ffill().astype(str).replace("nan", "")
        return df
    except Exception as exc:
        print(f"[excel] skip {path.name}: {exc}")
        return None


def clean_csv_df(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception as exc:
        print(f"[csv] skip {path.name}: {exc}")
        return None


def find_mssv_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if str(col).strip().lower() in MSSV_COLS:
            return col
    return None


def ingest_csv_exam_lists(db: Neo4jManager, csv_dir: Path) -> int:
    files = sorted(csv_dir.glob("*.csv"))
    total = 0
    print(f"[csv] importing {len(files)} files from {csv_dir}")
    for path in tqdm(files, desc="CSV"):
        df = clean_csv_df(path)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            msv = str(row.get("msv", row.get("mssv", ""))).strip()
            ngay_thi = str(row.get("ngay_thi", "")).strip()
            phong_thi = str(row.get("phong_thi", "")).strip()
            loai_thi = str(row.get("loai_thi", "")).strip()
            if not msv or msv == "nan":
                continue
            ki_thi_id = f"{row.get('ann_id', '')}_{ngay_thi}_{phong_thi}_{loai_thi}".replace(" ", "_")
            params = {
                "ann_id": str(row.get("ann_id", "")),
                "title": str(row.get("title", "")),
                "msv": msv,
                "ho_ten": str(row.get("ho_ten", "")),
                "ngay_sinh": str(row.get("ngay_sinh", "")),
                "loai_thi": loai_thi,
                "ngay_thi": ngay_thi,
                "gio_thi": str(row.get("gio_thi", "")),
                "phong_thi": phong_thi,
                "dia_diem": str(row.get("dia_diem_thi", "")),
                "ki_thi_id": ki_thi_id,
            }
            db.run(CYPHER_CSV, **params)
            total += 1
    return total


def ingest_excel_records(db: Neo4jManager, excel_dir: Path) -> int:
    files = sorted([path for path in excel_dir.iterdir() if path.suffix.lower() in {".xlsx", ".xls"}])
    total = 0
    print(f"[excel] importing {len(files)} files from {excel_dir}")
    for path in tqdm(files, desc="Excel"):
        ann_match = re.search(r"(ANN_[A-Z0-9]+)", path.name)
        ann_id = ann_match.group(1) if ann_match else "UNKNOWN"
        df = clean_excel_df(path)
        if df is None or df.empty:
            continue
        mssv_col = find_mssv_col(df)
        for idx, row in df.iterrows():
            props = {re.sub(r"[^a-zA-Z0-9_]", "_", str(key)): str(value) for key, value in row.items()}
            props["row_index"] = int(idx)
            props["raw_data"] = str(row.to_dict())
            props["ann_id"] = ann_id
            props["file_nguon"] = path.name
            db.run(CYPHER_RECORD, ann_id=ann_id, file_name=path.name, props=props)
            if mssv_col:
                mssv = str(row[mssv_col]).strip()
                if mssv and mssv != "nan":
                    db.run(CYPHER_RECORD_STUDENT, mssv=mssv, ann_id=ann_id)
            total += 1
    return total


def main() -> int:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    excel_dir = Path(args.excel_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV dir not found: {csv_dir}")
    if not excel_dir.exists():
        raise FileNotFoundError(f"Excel dir not found: {excel_dir}")

    db = Neo4jManager(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        if args.reset:
            print("[neo4j] resetting existing database...")
            db.reset()
        db.create_constraints()

        total_csv = ingest_csv_exam_lists(db, csv_dir)
        total_excel = ingest_excel_records(db, excel_dir)

        print()
        print("[done] import completed")
        print(f"  csv rows:   {total_csv}")
        print(f"  excel rows: {total_excel}")
        print(f"  total:      {total_csv + total_excel}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
