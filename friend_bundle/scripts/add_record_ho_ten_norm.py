"""
Migration: add ho_ten_norm to all Record nodes.
Parses raw_data (Python dict string), finds the name column,
applies vn_normalize(), sets rec.ho_ten_norm.
Run once. Safe to re-run (idempotent).
"""
import ast
import unicodedata
from neo4j import GraphDatabase

# ── diacritics normalizer (same as rag_service.py) ────────────────────────────
_VN_MAP = str.maketrans("đĐ", "dD")

def vn_normalize(s: str) -> str:
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return stripped.translate(_VN_MAP).lower()

# ── name column detection ─────────────────────────────────────────────────────
# After vn_normalize + strip spaces/underscores, these all become "hoten"
_NAME_KEYS_NORM = {
    "hoten", "hovaten", "hovten", "hoaten",
    "holoten", "hotensv", "namesv", "fullname",
}

def _key_sig(k: str) -> str:
    """Collapse a column key to a signature for matching."""
    return vn_normalize(str(k)).replace(" ", "").replace("_", "").replace("&", "")

def find_name(raw_data_str: str) -> str | None:
    try:
        d = ast.literal_eval(raw_data_str)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    # Priority: exact Vietnamese column names
    for col in ("Họ và Tên", "Họ và tên", "HỌ VÀ TÊN", "Họ Và Tên",
                "Họ tên", "HọTên", "Tên", "ho_ten", "Ho_ten", "HOTEN"):
        if col in d and isinstance(d[col], str) and d[col].strip():
            return d[col].strip()
    # Fallback: normalized key match
    for k, v in d.items():
        sig = _key_sig(k)
        if sig in _NAME_KEYS_NORM or "hoten" in sig:
            if isinstance(v, str) and len(v.strip()) > 1:
                return v.strip()
    return None

# ── Neo4j ─────────────────────────────────────────────────────────────────────
DRIVER = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"))
BATCH = 500

def run():
    with DRIVER.session() as s:
        total = s.run("MATCH (rec:Record) RETURN count(rec) AS n").single()["n"]
        print(f"Total Record nodes: {total}")

        # Fetch all in batches using SKIP/LIMIT
        updated = 0
        skipped = 0
        no_name = 0
        offset = 0

        while offset < total:
            rows = s.run(
                "MATCH (rec:Record) RETURN id(rec) AS rid, rec.raw_data AS raw SKIP $skip LIMIT $lim",
                skip=offset, lim=BATCH
            ).data()
            if not rows:
                break

            batch_updates = []
            for row in rows:
                raw = row["raw"]
                if not raw:
                    skipped += 1
                    continue
                name = find_name(raw)
                if not name:
                    no_name += 1
                    continue
                norm = vn_normalize(name)
                if norm:
                    batch_updates.append({"rid": row["rid"], "norm": norm})

            if batch_updates:
                s.run(
                    """
                    UNWIND $updates AS u
                    MATCH (rec:Record) WHERE id(rec) = u.rid
                    SET rec.ho_ten_norm = u.norm
                    """,
                    updates=batch_updates
                )
                updated += len(batch_updates)

            offset += BATCH
            if offset % 5000 == 0 or offset >= total:
                print(f"  Processed {min(offset, total)}/{total} — updated={updated}, no_name={no_name}")

        print(f"\nDone. Updated={updated}, no_name={no_name}, skipped(no raw_data)={skipped}")

        # Create index
        try:
            s.run("CREATE INDEX record_ho_ten_norm IF NOT EXISTS FOR (r:Record) ON (r.ho_ten_norm)")
            print("Index on Record.ho_ten_norm created (or already exists).")
        except Exception as e:
            print(f"Index creation note: {e}")

    DRIVER.close()

if __name__ == "__main__":
    run()
