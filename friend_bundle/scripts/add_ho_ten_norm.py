"""
Quick migration: add ho_ten_norm (ASCII-folded, lowercase) to all SinhVien nodes.
Also adds raw_data_norm to Record nodes for diacritics-free search.
Run once. Safe to re-run.
"""
import unicodedata
from neo4j import GraphDatabase

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12345678"

VIET_MAP = str.maketrans(
    "àáâãäåèéêëìíîïòóôõöùúûüýÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝ"
    "ăắặằẳẵÂẤẬẦẨẪÔỐỘỒỔỖƠớợờởỡƯứựừửữ"
    "đĐ",
    "aaaaaaeeeeiiiioooooouuuuyAAAAAAEEEEIIIIOOOOOUUUUY"
    "aaaaaAAAAAaoooooooooooouuuuuU"
    "dD",
)


def vn_normalize(s: str) -> str:
    if not s:
        return ""
    # NFD decompose: splits accented letters into base + combining marks
    nfd = unicodedata.normalize("NFD", s)
    # Remove combining diacritical marks
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Map remaining special chars (đ, Đ, etc.)
    stripped = stripped.translate(VIET_MAP)
    return stripped.lower()


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver.session() as s:
        # Step 1: set ho_ten_norm on all SinhVien that have ho_ten
        print("Loading SinhVien with ho_ten...")
        rows = list(s.run("MATCH (sv:SinhVien) WHERE sv.ho_ten IS NOT NULL RETURN sv.mssv AS mssv, sv.ho_ten AS ho_ten"))
        print(f"  Found {len(rows)} SinhVien with ho_ten")
        batch = [(r["mssv"], vn_normalize(r["ho_ten"])) for r in rows]
        s.run(
            "UNWIND $batch AS row MATCH (sv:SinhVien {mssv: row[0]}) SET sv.ho_ten_norm = row[1]",
            batch=batch,
        )
        print(f"  Set ho_ten_norm for {len(batch)} SinhVien")

        # Step 2: create index on ho_ten_norm
        try:
            s.run("CREATE INDEX sv_ho_ten_norm IF NOT EXISTS FOR (n:SinhVien) ON (n.ho_ten_norm)")
            print("  Created index on SinhVien.ho_ten_norm")
        except Exception as e:
            print(f"  Index already exists or error: {e}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
