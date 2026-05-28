"""One-time migration: set titles for Excel Announcements + set ho_ten on SinhVien from Record data."""
from __future__ import annotations
import ast
from neo4j import GraphDatabase

TITLES = {
    "ANN_03EAC84BDC": "Danh sach sinh vien chua nop phi BHYT dot 3 nam 2025-2026",
    "ANN_0DB422D1EA": "Danh sach lop ghep hoc ky 2 nam hoc 2025-2026",
    "ANN_16EA450F0A": "Lich sinh hoat cong dan sinh vien cuoi khoa K2021",
    "ANN_1D371CB2BD": "Danh sach sinh vien giao Do an tot nghiep ky 2 nam hoc 2025-2026 (V2)",
    "ANN_2727A94190": "Danh sach sinh vien phan xe hoc GDQP dot 3 dieu chinh",
    "ANN_2BDBC54CBF": "Hoan tra tien GDQP K2025 dot 1",
    "ANN_2E5F94A628": "Danh sach sinh vien ky tuc xa hoc ky II nam hoc 2025-2026",
    "ANN_36998292DA": "Danh sach sinh vien thua van bang de nghi tra 01/03/2024",
    "ANN_43D38F5DEA": "Danh sach BHYT nam 2025 (phu luc 1, 2, 3)",
    "ANN_443E937C04": "Danh sach sinh vien hoan tra hoc phi nam hoc 2024-2025",
    "ANN_47C34DDF11": "Danh sach sinh vien thoi hoc va canh bao hoc vu HK2 2024-2025",
    "ANN_49BB55F3FA": "Danh sach sinh vien nhan the BHTT nam hoc 2025-2026 dot 2",
    "ANN_5F4C6D4386": "Danh sach sinh vien K41 hoc GDQP dot 2 (DUT) chia xe",
    "ANN_668F749D94": "Danh sach sinh vien bo sung mon Thuc nghiem Vat ly thang 12/2025",
    "ANN_6B3752A79B": "Danh sach sinh vien dang ky hoc phan qua so tin chi 2026",
    "ANN_6C2862BE2D": "Danh sach sinh vien hoan tra BHYT K2025 IV (chua co the)",
    "ANN_6C3B8659FA": "Danh sach sinh vien phai nop va chua cap nhat thong tin BHYT lan 4",
    "ANN_6F4BB1C813": "Danh sach lop mo tang cuong hoc ky 2520",
    "ANN_7C3A741E63": "Quyet dinh mien hoc tieng Anh/Phap K2025 HK2 (465 SV)",
    "ANN_854ABBB4B9": "Thoi khoa bieu tang cuong dot 1 nam hoc 2025-2026",
    "ANN_8BBC342A89": "Danh sach sinh vien ban dang ky nhan bang tot nghiep thang 8/2025",
    "ANN_8F6F9E7DB7": "Danh sach sinh vien hoan tra hoc phi HK1 qua HK2 nam hoc 2025-2026",
    "ANN_947D54D31B": "Danh sach sinh vien GDQP dot 1 K2025",
    "ANN_9E11F72C4E": "Danh sach sinh vien du kien tot nghiep thang 12 nam 2025 (V1)",
    "ANN_9E93064DD4": "Danh sach sinh vien nop hoc phi hoan xet tot nghiep T5,T8,T9 nam 2025",
    "ANN_A66E88D914": "Danh sach sinh vien nhan de tai DATN ky 2 nam hoc 2025-2026 (V4)",
    "ANN_A6FF8F21A6": "Thoi khoa bieu tang cuong dot 2 nam hoc 2025-2026",
    "ANN_AA62548CB0": "Danh sach sinh vien phan xe hoc GDQP dot 3",
    "ANN_AB60A25FAB": "Danh sach sinh vien hoc phi hoc ky He nam 2024-2025",
    "ANN_ACB6B3017B": "Danh sach sinh vien TN GHP mien giam hoc phi KHI 2025-2026",
    "ANN_B4624B0244": "Danh sach sinh vien hoan tra hoc phi HK1 2025-2026 (K20, K25)",
    "ANN_B53B596634": "Danh sach sinh vien du kien xet tot nghiep dot 1 nam 2026 (V2)",
    "ANN_BACAB23551": "Danh sach sinh vien tot nghiep thang 12 nam 2025",
    "ANN_BE1800B4EA": "Danh sach sinh vien dang ky chot K41",
    "ANN_C1903D7D2A": "Danh sach lop hoc phan huy hoc ky 2520",
    "ANN_C2F7BCC111": "Danh sach sinh vien giao Do an tot nghiep ky 2 nam hoc 2025-2026 (V3)",
    "ANN_C3AF778C7B": "Danh sach sinh vien thoi hoc va canh bao hoc vu HK2 2025-2026",
    "ANN_C7F7E35644": "Danh sach sinh vien nhan the BHTT nam hoc 2025-2026 dot 1",
    "ANN_CE6CB26747": "Thoi khoa bieu hoc ky 2 nam hoc 2025-2026",
    "ANN_D4CB437F15": "Danh sach thu tu trao bang tot nghiep",
    "ANN_D89F7F7A92": "Danh sach sinh vien du kien xet tot nghiep dot 1 nam 2026 (V3)",
    "ANN_DAA0B018AE": "Danh sach lop mo bo sung hoc ky 2 nam hoc 2025-2026",
    "ANN_E2E04C0D54": "Danh sach sinh vien dang ky BHTT dot 2 (24/11/2023)",
    "ANN_E9BF08EA15": "Danh sach sinh vien du kien xet tot nghiep dot 1 nam 2026 (V1)",
    "ANN_EB28704715": "Danh sach sinh vien no tien hoc phi xet tot nghiep dot 1/2024",
    "ANN_ECFA9D95A5": "Danh sach sinh vien nhan de tai DATN ky 2 nam hoc 2025-2026 (V1)",
    "ANN_F29F81BD31": "Danh sach sinh vien chua tham gia BHYT nam 2024 dot 3",
    "ANN_FC267A7C10": "Danh sach sinh vien chua hoan thanh hoc phi xet tot nghiep thang 12/2025",
}

# Vietnamese name column names (after pandas sanitization: non-alphanum → _)
NAME_COLS = {"Hoten", "Ho_ten", "Ho_va_ten", "Hovaten", "HO_TEN", "Ho_Ten",
             "H_v_t_n_ng_i_h_c", "H_v_t_n", "Hoten_nguoi_hoc", "ten_sv",
             "HOTEN", "ho_ten", "TEN", "Ten", "HoTen"}

# MSSV column names after sanitization
MSSV_COLS = {"SotheSV", "mssv", "MaSV", "MSSV", "Masv", "MaHS", "mahs",
             "SoTheSV", "msv", "Ma_sv", "Ma_SV", "STH_SV"}


def find_col_value(d: dict, col_set: set) -> str:
    for k, v in d.items():
        if k in col_set:
            return str(v).strip()
    return ""


def main():
    driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"))
    with driver.session() as s:
        # Step 1: Set Announcement titles
        print("=== Step 1: Setting Announcement titles ===")
        updated = 0
        for ann_id, title in TITLES.items():
            result = s.run(
                "MATCH (a:Announcement {id: $aid}) WHERE a.title IS NULL SET a.title = $t RETURN a.id",
                aid=ann_id, t=title
            )
            if result.single():
                updated += 1
        r = s.run("MATCH (a:Announcement) RETURN count(a) as tot, count(a.title) as with_title")
        print(f"  Updated {updated} titles. Total: {dict(r.single())}")

        # Step 2: Set ho_ten on SinhVien from Record.raw_data
        print("=== Step 2: Setting ho_ten on SinhVien from Record data ===")
        # Get all Records that have raw_data with name info
        batch_size = 500
        skip = 0
        total_updated = 0
        while True:
            rows = list(s.run(
                "MATCH (rec:Record) WHERE rec.raw_data IS NOT NULL "
                "RETURN rec.raw_data AS rd, rec.ann_id AS aid SKIP $skip LIMIT $lim",
                skip=skip, lim=batch_size
            ))
            if not rows:
                break
            skip += batch_size
            for row in rows:
                raw = row["rd"]
                ann_id = row["aid"]
                if not raw or raw == "None":
                    continue
                try:
                    d = ast.literal_eval(raw)
                except Exception:
                    continue
                mssv = find_col_value(d, MSSV_COLS)
                name = find_col_value(d, NAME_COLS)
                if not mssv or not name or mssv in ("nan", "") or name in ("nan", ""):
                    continue
                # Only set ho_ten if SinhVien exists but has no ho_ten
                result = s.run(
                    "MATCH (sv:SinhVien {mssv: $mssv}) WHERE sv.ho_ten IS NULL "
                    "SET sv.ho_ten = $name RETURN sv.mssv",
                    mssv=mssv, name=name
                )
                if result.single():
                    total_updated += 1
        r = s.run("MATCH (sv:SinhVien) RETURN count(sv) as tot, count(sv.ho_ten) as with_name")
        print(f"  Set ho_ten for {total_updated} SinhVien. Total: {dict(r.single())}")

        # Step 3: Create full-text index on Record.raw_data for fast name search
        print("=== Step 3: Creating full-text index ===")
        try:
            s.run("CREATE FULLTEXT INDEX record_rawdata_idx IF NOT EXISTS FOR (r:Record) ON EACH [r.raw_data]")
            print("  Full-text index created (or already exists)")
        except Exception as e:
            print(f"  Index creation note: {e}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
